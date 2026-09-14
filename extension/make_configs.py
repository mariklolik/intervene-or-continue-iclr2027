import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.task_sampling import SCIENCEWORLD_NONDETERMINISTIC, alfworld_tasks

ENVIRONMENTS = ("alfworld", "scienceworld")
PHASE_OFFSETS = {"dev": 1, "test": 2}
ALFWORLD_SPLITS = {"dev": "valid_seen", "test": "valid_unseen"}
TEMPLATE_FIELDS = {"phase", "model", "temperature", "rounds", "arms", "tasks", "scaffold", "max_tokens", "system_prompt", "schema_version"}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_document(path: Path) -> tuple[dict, str]:
    content = Path(path).read_bytes()
    value = json.loads(content)
    if not isinstance(value, dict) or value.get("schema_version", 1) != 1:
        raise ValueError(f"Unsupported schema in {path}")
    return value, hashlib.sha256(content).hexdigest()


def validate_template(value: dict, actor: bool) -> None:
    required = {"phase", "model", "temperature", "rounds", "arms", "tasks"}
    if actor:
        required |= {"scaffold", "max_tokens", "system_prompt"}
    if not required <= value.keys() or value.keys() - TEMPLATE_FIELDS:
        raise ValueError("Unsupported template schema")
    if not isinstance(value["tasks"], list) or not isinstance(value["arms"], dict):
        raise ValueError("Invalid template tasks or arms")
    if actor and (value["scaffold"] != "grounded-react-v1" or value["max_tokens"] != 256 or not isinstance(value["system_prompt"], str) or not value["system_prompt"]):
        raise ValueError("Unsupported actor interface")
    if not actor and (set(value["arms"]) != {"A0", "A1", "A2", "A3"} or value["arms"]["A0"] is not None or any(not isinstance(value["arms"][arm], str) or not value["arms"][arm] for arm in ("A1", "A2", "A3"))):
        raise ValueError("A0-A3 intervention contract required")
    if type(value["rounds"]) is not int or value["rounds"] < int(not actor):
        raise ValueError("Invalid replica count")


def scienceworld_pools(document: dict) -> dict:
    catalog = document.get("catalog")
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("ScienceWorld catalogue missing")
    pools = {}
    for family, row in sorted(catalog.items()):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid family schema: {family}")
        ids = row.get("ids", {})
        if set(ids) != {"train", "dev", "test"}:
            raise ValueError(f"Invalid split schema: {family}")
        for split, values in ids.items():
            if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values) or len(values) != len(set(values)) or row.get("counts", {}).get(split) != len(values):
                raise ValueError(f"Invalid variation pool: {family}/{split}")
        if any(set(ids[left]) & set(ids[right]) for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))):
            raise ValueError(f"ScienceWorld split overlap: {family}")
        if family not in SCIENCEWORLD_NONDETERMINISTIC:
            pools[family] = ids
    return pools


def sample_scienceworld(pools: dict, phase: str, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    families = list(pools)
    rng.shuffle(families)
    variations = {family: list(pools[family][phase]) for family in families}
    for values in variations.values():
        rng.shuffle(values)
    selected = []
    for index in range(max((len(values) for values in variations.values()), default=0)):
        for family in families:
            if len(selected) == count:
                return selected
            if index < len(variations[family]):
                selected.append({"task_name": family, "variation": variations[family][index], "simplification": "", "task_type": family})
    return selected


def select_tasks(data_root: Path, pools: dict, phase: str, counts: dict) -> list[dict]:
    seed = 260908 + PHASE_OFFSETS[phase]
    previous = os.environ.get("ALFWORLD_DATA")
    os.environ["ALFWORLD_DATA"] = str(data_root)
    try:
        specifications = {
            "alfworld": alfworld_tasks(counts["alfworld"], ALFWORLD_SPLITS[phase], seed),
            "scienceworld": sample_scienceworld(pools, phase, counts["scienceworld"], seed),
        }
    finally:
        if previous is None:
            os.environ.pop("ALFWORLD_DATA", None)
        else:
            os.environ["ALFWORLD_DATA"] = previous
    tasks = []
    offset = 0
    for env in ENVIRONMENTS:
        if len(specifications[env]) != counts[env]:
            raise ValueError(f"{phase}/{env} shortfall: requested {counts[env]}, available {len(specifications[env])}")
        group = []
        for spec in specifications[env]:
            if env == "alfworld":
                relative = Path(spec["game_file"]).relative_to(data_root / "json_2.1.1" / ALFWORLD_SPLITS[phase])
                if len(relative.parts) != 3 or relative.name != "game.tw-pddl":
                    raise ValueError("Unsupported ALFWorld task identity")
                identity = {"game": "/".join(relative.parts[:2])}
            else:
                identity = {key: spec[key] for key in ("task_name", "variation", "simplification")}
            task_id = digest({"env": env, "identity": identity})[:16]
            choice = int(digest([task_id, "checkpoint"])[0], 16) % 2
            group.append({
                "task_id": task_id, "split": phase, "env": env,
                "task_spec": spec, "identity": identity,
                "source_split": ALFWORLD_SPLITS[phase] if env == "alfworld" else phase,
                "checkpoint_step": ((4, 8) if env == "alfworld" else (8, 16))[choice],
                "seed": int(digest([seed, task_id, "actor_seed"])[:8], 16) % 2147483647,
            })
        group.sort(key=lambda task: digest([task["task_id"], "shard"]))
        for index, task in enumerate(group):
            task["worker_shard"] = (index + offset) % 2
        offset = (offset + len(group)) % 2
        tasks.extend(group)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError(f"Duplicate task identities in {phase}")
    random.Random(seed).shuffle(tasks)
    return tasks


def build_configs(base_paths: list[Path], arms_path: Path, readiness_path: Path, scienceworld_path: Path, data_root: Path, counts: dict) -> dict[str, dict]:
    if set(counts) != set(PHASE_OFFSETS) or any(set(row) != set(ENVIRONMENTS) for row in counts.values()):
        raise ValueError("Counts require dev/test and both environments")
    if any(type(count) is not int or count < 0 for row in counts.values() for count in row.values()) or any(sum(row.values()) == 0 for row in counts.values()):
        raise ValueError("Nonnegative counts and nonempty phases required")
    arms, arms_hash = read_document(arms_path)
    readiness, readiness_hash = read_document(readiness_path)
    scienceworld, scienceworld_hash = read_document(scienceworld_path)
    validate_template(arms, False)
    pools = scienceworld_pools(scienceworld)
    try:
        image = readiness["sglang"]["image"]
        models = readiness["models"]
        packages = readiness["cpu"]["packages"]
        datasets = {
            "alfworld": {
                "version": "json_2.1.1", "package_version": packages["alfworld"],
                "textworld_version": packages["textworld"], "assets": readiness["assets"],
            },
            "scienceworld": {
                "version": scienceworld["scienceworld_version"],
                "source_sha256": scienceworld["source_sha256"],
            },
        }
        verified = scienceworld["probe"]["passed"] is True
    except (KeyError, TypeError) as error:
        raise ValueError("Unsupported readiness schema") from error
    if not verified or not isinstance(image, str) or not re.search(r"@sha256:[0-9a-f]{64}$", image):
        raise ValueError("Verified ScienceWorld and immutable image required")
    hashes = [asset.get("sha256", "") for asset in datasets["alfworld"]["assets"]]
    hashes.extend(datasets["scienceworld"]["source_sha256"].values())
    if not datasets["alfworld"]["assets"] or not datasets["scienceworld"]["source_sha256"] or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
        raise ValueError("Dataset hashes required")
    templates = []
    for path in base_paths:
        base, base_hash = read_document(path)
        validate_template(base, True)
        matches = [row for name, row in models.items() if name.rsplit("/", 1)[-1] == base["model"]]
        if len(matches) != 1 or matches[0].get("all_indexed_weights_present") is not True:
            raise ValueError(f"Unknown or unverified model: {base['model']}")
        model_path = Path(matches[0].get("path", ""))
        if model_path.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", model_path.name):
            raise ValueError("Exact model revision required")
        templates.append((base, base_hash, str(model_path)))
    if not templates or len({base["model"] for base, _, _ in templates}) != len(templates):
        raise ValueError("Distinct actor templates required")
    interface = ("temperature", "scaffold", "max_tokens", "system_prompt")
    if any(any(base[key] != templates[0][0][key] for key in interface) for base, _, _ in templates):
        raise ValueError("Actor scaffolds must match")
    selected = {phase: select_tasks(Path(data_root), pools, phase, row) for phase, row in counts.items()}
    if {task["task_id"] for task in selected["dev"]} & {task["task_id"] for task in selected["test"]}:
        raise ValueError("Dev/test task identity overlap")
    configs = {}
    for base, base_hash, model_path in templates:
        for phase, tasks in selected.items():
            common = {key: base[key] for key in ("model", *interface)}
            common.update({
                "schema_version": 1, "phase": phase,
                "rounds": arms["rounds"], "arms": arms["arms"],
                "model_path": model_path, "model_revision": Path(model_path).name,
                "image": image, "datasets": datasets,
                "sampling": {
                    "seed": 260908 + PHASE_OFFSETS[phase],
                    "requested_counts": counts[phase], "task_set_sha256": digest(tasks),
                },
                "provenance": {
                    "base_config_sha256": base_hash, "arms_config_sha256": arms_hash,
                    "readiness_sha256": readiness_hash,
                    "scienceworld_readiness_sha256": scienceworld_hash,
                    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "sampler_sha256": hashlib.sha256(Path(alfworld_tasks.__code__.co_filename).read_bytes()).hexdigest(),
                },
                "worker_shards": 2,
            })
            for shard in (None, 0, 1):
                config = copy.deepcopy(common)
                config["worker_shard"] = shard
                config["tasks"] = copy.deepcopy([task for task in tasks if shard is None or task["worker_shard"] == shard])
                suffix = "" if shard is None else f"-shard{shard}"
                configs[f"main-{phase}-{base['model']}{suffix}.json"] = config
    return configs


def write_configs(configs: dict[str, dict], output: Path) -> None:
    if any((output / name).exists() for name in configs):
        raise FileExistsError("Frozen configuration already exists")
    output.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        with (output / name).open("x") as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, action="append", required=True)
    parser.add_argument("--arms-config", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--scienceworld-readiness", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    for phase in PHASE_OFFSETS:
        for env in ENVIRONMENTS:
            parser.add_argument(f"--{phase}-{env}", type=int, required=True)
    args = parser.parse_args()
    counts = {phase: {env: getattr(args, f"{phase}_{env}") for env in ENVIRONMENTS} for phase in PHASE_OFFSETS}
    try:
        configs = build_configs(args.base_config, args.arms_config, args.readiness, args.scienceworld_readiness, args.alfworld_data, counts)
        write_configs(configs, args.out)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.out), "files": sorted(configs), "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
