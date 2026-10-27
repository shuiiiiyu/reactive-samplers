"""Prepare one paired exploratory 10-COCO, two-model fixed-grid experiment."""
import argparse
import json
from pathlib import Path
import random

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-root', type=Path, default=ROOT)
    parser.add_argument('--name', default='coco10-grid-20261027', help='New directory name under sampler_runs/')
    args = parser.parse_args()
    relative = Path('sampler_runs') / args.name
    output, runtime = ROOT / relative, args.runtime_root / relative
    output.mkdir(parents=True, exist_ok=False)
    old = set()
    for path in (ROOT / 'sampler_runs/datasets/coco_hpd_20_v1').glob('*.jsonl'):
        old.update(json.loads(line)['prompt'] for line in path.read_text().splitlines() if line.strip())
    annotations = json.loads((ROOT / 'benchmarks/coco/data/annotations/captions_train2017.json').read_text())['annotations']
    rng = random.Random(20261027)
    rng.shuffle(annotations)
    rows, images, captions = [], set(), set()
    for item in annotations:
        prompt = item['caption'].strip()
        if prompt and prompt not in old and prompt not in captions and item['image_id'] not in images:
            images.add(item['image_id'])
            captions.add(prompt)
            rows.append({'id': f'coco10_{len(rows):02d}', 'source': 'coco_train2017', 'prompt': prompt,
                         'seed': rng.randrange(2**32), 'image_id': item['image_id'], 'annotation_id': item['id']})
        if len(rows) == 10:
            break
    assert len(rows) == 10
    (output / 'prompts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (output / 'experiment.json').write_text(json.dumps({'title': 'COCO 十样本跳步对比实验 · SD3 与 SDXL',
        'selection_seed': 20261027, 'count': 10, 'split': 'exploratory paired comparison',
        'agent_model': 'gpt-6-luna', 'reasoning_effort': 'low', 'service_tier': 'default',
        'fast_mode': False, 'previous_prompt_sets_excluded': True}, ensure_ascii=False, indent=2))
    policy = output / 'keep_all'
    policy.mkdir()
    (policy / 'analysis.py').write_text('FEATURE_LAYERS=[]\nHISTORY_LIMIT=0\ndef analyze(ctx,memory): return {}\n')
    (policy / 'updater.py').write_text('from reactive_samplers.context import SkipAction\ndef choose(obs,ctx,memory): return SkipAction(False)\n')
    for model in ('sd3', 'sdxl'):
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'conf')):
            config = OmegaConf.to_container(compose(config_name=f'{model}_main'), resolve=True)
        config['model_path'] = str(args.runtime_root / config['model_path'])
        for arm in ('default', 'skip'):
            arm_config = {**config, 'method': 'grid_skip', 'mode': 'default' if arm == 'default' else 'explore',
                'expected_count': 10,
                'prompts': str(runtime / 'prompts.jsonl'), 'output_dir': str(runtime / f'{model}_{arm}'),
                'policy': str(runtime / 'keep_all') if arm == 'default' else None,
                'baseline_results': str(runtime / f'{model}_default/results.json') if arm == 'skip' else None,
                'metrics': {'names': ['hps_v2_1', 'clip_score'],
                    'clip_model': str(args.runtime_root / 'metrics/clip/checkpoints/ViT-B-32.pt'),
                    'hps_checkpoint': str(args.runtime_root / 'benchmarks/hps/checkpoints/hpsv2/HPS_v2.1_compressed.pt')}}
            (output / f'{model}_{arm}.json').write_text(json.dumps(arm_config, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
