import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path):
    print('>>>', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    ap = argparse.ArgumentParser(description='Run full OCR pipeline end-to-end')
    ap.add_argument('--labelstudio_json', type=str, required=True)
    ap.add_argument('--images_dir', type=str, required=True)
    ap.add_argument('--output_base', type=str, required=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = Path(args.output_base)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1) Analyze
    run(['python', str(root / 'data' / 'analyze_dataset.py'), '--json_path', args.labelstudio_json, '--output_dir', str(data_dir / 'analysis')], cwd=root)

    # 2) Convert (detection+recognition) + splits
    run(['python', str(root / 'data' / 'convert_dataset.py'), '--labelstudio_json', args.labelstudio_json, '--images_dir', args.images_dir, '--output_dir', str(data_dir), '--vocab_path', str(data_dir / 'vocab.json')], cwd=root)

    # 3) Preprocess images (original/clahe/highboost)
    run(['python', str(root / 'data' / 'preprocess.py'), '--input_dir', args.images_dir, '--output_base', str(data_dir / 'preproc')], cwd=root)

    # 4) Train detection (3 variants placeholder — same config repeated)
    run(['python', str(root / 'training' / 'train_detection.py'), '--data_dir', str(data_dir), '--output_name', 'det_original'], cwd=root)
    run(['python', str(root / 'training' / 'train_detection.py'), '--data_dir', str(data_dir), '--output_name', 'det_clahe'], cwd=root)
    run(['python', str(root / 'training' / 'train_detection.py'), '--data_dir', str(data_dir), '--output_name', 'det_highboost'], cwd=root)

    # 5) Train recognition (3 variants placeholder) — assumes recognition splits already created
    run(['python', str(root / 'training' / 'train_recognition.py'), '--data_dir', str(data_dir), '--vocab_path', str(data_dir / 'vocab.json'), '--output_name', 'rec_original'], cwd=root)
    run(['python', str(root / 'training' / 'train_recognition.py'), '--data_dir', str(data_dir), '--vocab_path', str(data_dir / 'vocab.json'), '--output_name', 'rec_clahe'], cwd=root)
    run(['python', str(root / 'training' / 'train_recognition.py'), '--data_dir', str(data_dir), '--vocab_path', str(data_dir / 'vocab.json'), '--output_name', 'rec_highboost'], cwd=root)

    # 6) End-to-end evaluation over 9 combos
    run(['python', str(root / 'evaluation' / 'evaluate_end_to_end.py'), '--test_data', str(data_dir / 'detection_test.json'), '--models_dir', str(data_dir / 'runs'), '--vocab_path', str(data_dir / 'vocab.json'), '--output_dir', str(data_dir / 'eval')], cwd=root)

    # 7) Generate results tables/plots
    run(['python', str(root / 'evaluation' / 'generate_results.py'), '--logs_dir', str(data_dir / 'runs'), '--eval_dir', str(data_dir / 'eval'), '--output_dir', str(data_dir / 'results')], cwd=root)

    print('Full pipeline complete. Outputs at', data_dir)


if __name__ == '__main__':
    main()
