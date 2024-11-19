import csv

# 가짜 데이터 생성
all_params = [
    {"param1": 0.1, "param2": 0.2},
    {"param1": 0.3, "param2": 0.4},
    {"param1": 0.5, "param2": 0.6},
]

all_metrics = {
    "Accuracy": [0.95, 0.96, 0.94],
    "Loss": [0.1, 0.08, 0.09],
    "Precision": [0.93, 0.92, 0.91],
}

# 실험 결과를 CSV 파일로 저장
csv_filename = "grid_search_exp_results.csv"
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # 헤더 작성
    writer.writerow(["Metric"] + [f"Trial {i+1}" for i in range(len(all_params))])

    # 각 지표의 결과 작성
    for metric, values in all_metrics.items():
        writer.writerow([metric] + [f"{v:.4f}" for v in values])

print(f"Experiment results saved to: {csv_filename}")
