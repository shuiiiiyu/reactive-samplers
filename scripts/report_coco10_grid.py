"""Paired quality/speed report and exact timestep/Agent-output audit, no model calls."""
import argparse
import csv
import io
import json
from pathlib import Path
from statistics import mean


def report(root):
    arms, traces, pairs = {}, [], []
    for model in ('sd3', 'sdxl'):
        for arm in ('default', 'skip'):
            folder = root / f'{model}_{arm}'
            path = folder / 'results.json'
            arms[f'{model}_{arm}'] = json.loads(path.read_text())['images'] if path.exists() else []
            for row in arms[f'{model}_{arm}']:
                events = folder / 'trajectories' / row['id'] / 'events.jsonl'
                for line in events.read_text().splitlines():
                    event = json.loads(line)
                    if event['event'] != 'step':
                        continue
                    number = event.get('agent_decision')
                    answer = folder / 'decisions' / f'{number:05d}' / 'answer.txt' if number is not None else None
                    traces.append({'model': model, 'arm': arm, 'sample_id': row['id'], 'seed': row['seed'],
                        **event, 'agent_output_text': answer.read_text() if answer and answer.exists() else None})
        reference = {(r['prompt'], r['seed']): r for r in arms[f'{model}_default']}
        for row in arms[f'{model}_skip']:
            base = reference.get((row['prompt'], row['seed']))
            if base is None:
                continue
            pairs.append({'model': model, 'sample_id': row['id'], 'prompt': row['prompt'], 'seed': row['seed'],
                'baseline_steps': base['steps'], 'skip_steps': row['steps'],
                'baseline_nfe': base['nfe'], 'skip_nfe': row['nfe'],
                'nfe_reduction': 1 - row['nfe'] / base['nfe'],
                'hps_delta': row['scores']['hps_v2_1'] - base['scores']['hps_v2_1'],
                'clip_delta': row['scores']['clip_score'] - base['scores']['clip_score'],
                'baseline_seconds': base['end_to_end_seconds'], 'skip_seconds': row['end_to_end_seconds'],
                'wall_speedup': base['end_to_end_seconds'] / row['end_to_end_seconds'],
                'baseline_denoiser_seconds': base['denoiser_seconds'], 'skip_denoiser_seconds': row['denoiser_seconds'],
                'denoiser_speedup': base['denoiser_seconds'] / row['denoiser_seconds'],
                'agent_seconds': row['agent_seconds'], 'agent_calls': row['agent_calls'],
                'valid_pair': base['valid_for_benchmark'] and row['valid_for_benchmark'],
                'baseline_timesteps': json.dumps(base['timesteps']), 'skip_timesteps': json.dumps(row['timesteps']),
                'kept_indices': json.dumps(row['kept_indices']), 'skipped_indices': json.dumps(row['skipped_indices'])})
    (root / 'decision_trace.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n')
    if pairs:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
        (root / 'comparison.csv').write_text(stream.getvalue(), encoding='utf-8-sig')
    lines = ['# COCO 十样本跳步对比实验', '',
        '同一批 COCO Train prompt 与 seed；Luna / low / 标准速度。属于在线探索对照，不能作为独立冻结泛化评测。',
        '总耗时包含 Agent 等待；模型前向耗时单独记录。回退样本保留并明确标记，不能混入干净方法对比。', '',
        '| 组别 | 已完成 | 平均 HPSv2.1 | 平均 CLIPScore | 平均 NFE | 平均前向秒数 | 平均总秒数 | 平均 Agent 秒数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, rows in arms.items():
        if rows:
            values = [mean(r['scores']['hps_v2_1'] for r in rows), mean(r['scores']['clip_score'] for r in rows),
                mean(r['nfe'] for r in rows), mean(r['denoiser_seconds'] for r in rows),
                mean(r['end_to_end_seconds'] for r in rows), mean(r['agent_seconds'] for r in rows)]
            lines.append(f'| {name} | {len(rows)}/10 | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
        else:
            lines.append(f'| {name} | 0/10 | — | — | — | — | — | — |')
    for model in ('sd3', 'sdxl'):
        all_pairs = [p for p in pairs if p['model'] == model]
        clean = [p for p in all_pairs if p['valid_pair']]
        lines += ['', f'## {model.upper()} 配对结果', '', f'完整配对 {len(all_pairs)}/10；无失败/回退的配对 {len(clean)}/10。']
        if clean:
            lines.append('仅对无失败/回退配对统计：' + '；'.join(
                f'{label} {mean(p[key] for p in clean):.4f}' for key, label in (
                    ('nfe_reduction', 'NFE 减少比例'), ('hps_delta', 'HPS 差值'), ('clip_delta', 'CLIPScore 差值'),
                    ('denoiser_speedup', '前向加速比'), ('wall_speedup', '包含 Agent 的总耗时加速比'))) + '。')
    lines += ['', '逐图配置、timestep 和保留/跳过索引见 comparison.csv；逐步 action、策略版本、Agent 最终输出见 decision_trace.json。',
              'Agent 输出是该次代码修改/保留的原文说明；实际逐候选 action 以 grid_decisions 为准。', '']
    (root / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({name: len(rows) for name, rows in arms.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    report(parser.parse_args().root)
