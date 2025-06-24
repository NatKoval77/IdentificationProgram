import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
import seaborn as sns

def load_latest_results():
    print("Выберите тестирование:")
    print("0 - Результаты базовой модели")
    print("1 - Результаты дообученной модели")
    
    while True:
        try:
            choice = int(input("Введите 0 или 1: "))
            if choice in [0, 1]:
                break
            else:
                print("Пожалуйста, введите 0 или 1")
        except ValueError:
            print("Пожалуйста, введите число 0 или 1")
    
    model_suffix = "finetuned" if choice == 1 else "base"
    result_files = glob.glob(f'test_results_{model_suffix}_*.csv')
    if not result_files:
        raise FileNotFoundError(f"Не найдены файлы с результатами тестов для {'дообученной' if choice == 1 else 'базовой'} модели")
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"Загружаем результаты из {latest_file}")
    return pd.read_csv(latest_file)

def analyze_distances(df):
    pair_results = df[df['test_type'].str.contains('pair')].copy()
    
    positive_pairs = pair_results[pair_results['test_type'] == 'positive_pair']
    negative_pairs = pair_results[pair_results['test_type'] == 'negative_pair']
    
    stats = {}
    for model in df['model'].unique():
        model_pos = positive_pairs[positive_pairs['model'] == model]
        model_neg = negative_pairs[negative_pairs['model'] == model]
        
        stats[model] = {
            'positive_mean': model_pos['distance'].mean(),
            'positive_std': model_pos['distance'].std(),
            'negative_mean': model_neg['distance'].mean(),
            'negative_std': model_neg['distance'].std()
        }
    
    return stats

def calculate_metrics(df):
    metrics = {}
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        time_ms = model_data['time'].mean() * 1000
        frontal_acc = model_data[model_data['test_type'] == 'frontal']['detected'].mean() * 100
        pairs_acc = model_data[model_data['test_type'].str.contains('pair')]['detected'].mean() * 100
        challenge_types = {
            'ac': 'accessories',
            'gl': 'glasses',
            'ms': 'mask',
            'bad': 'bad_quality'
        }
        
        challenge_acc = {}
        for code, name in challenge_types.items():
            challenge_data = model_data[model_data['test_type'].str.contains(name)]
            if not challenge_data.empty:
                challenge_acc[name] = challenge_data['detected'].mean() * 100
            else:
                challenge_acc[name] = 0.0
        
        challenge_total = model_data[model_data['test_type'].str.contains('challenge')]['detected'].mean() * 100
        
        metrics[model] = {
            'time_ms': time_ms,
            'frontal_acc': frontal_acc,
            'pairs_acc': pairs_acc,
            **challenge_acc,
            'challenge_total': challenge_total
        }
    
    return metrics

def create_bar_charts(metrics_df, distance_stats, report_dir):
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    speed_data = metrics_df[['Модель', 'Время обработки (мс)']].set_index('Модель')
    speed_data['Время обработки (мс)'] = speed_data['Время обработки (мс)'].astype(float)
    bars = speed_data.plot(kind='bar', ax=axes[0], title='Среднее время обработки (мс)')
    axes[0].set_ylabel('Время (мс)')
    axes[0].tick_params(axis='x', rotation=45)
    for bar in bars.patches:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
    detection_data = metrics_df[['Модель', 'Точность (front, %)', 'Точность (pairs, %)']].set_index('Модель')
    detection_data = detection_data.astype(float)
    bars = detection_data.plot(kind='bar', ax=axes[1], title='Точность детекции (%)')
    axes[1].set_ylabel('Процент успешных детекций')
    axes[1].tick_params(axis='x', rotation=45)
    for bar in bars.patches:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
    models = list(distance_stats.keys())
    positive_means = [stats['positive_mean'] for stats in distance_stats.values()]
    negative_means = [stats['negative_mean'] for stats in distance_stats.values()]
    
    x = range(len(models))
    width = 0.35
    
    pos_bars = axes[2].bar([i - width/2 for i in x], positive_means, width, label='Позитивные пары')
    neg_bars = axes[2].bar([i + width/2 for i in x], negative_means, width, label='Негативные пары')

    for bar in pos_bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
    
    for bar in neg_bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
    
    axes[2].set_title('Средние расстояния между эмбеддингами')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45)
    axes[2].set_ylabel('Евклидово расстояние')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{report_dir}/bar_charts.png')
    plt.close()

def generate_report():
    results_df = load_latest_results()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = f'report_{timestamp}'
    os.makedirs(report_dir, exist_ok=True)
    metrics = calculate_metrics(results_df)
    
    table_data = []
    for model, model_metrics in metrics.items():
        row = {
            'Модель': model,
            'Время обработки (мс)': f"{model_metrics['time_ms']:.1f}",
            'Точность (front, %)': f"{model_metrics['frontal_acc']:.1f}",
            'Точность (pairs, %)': f"{model_metrics['pairs_acc']:.1f}",
            'Аксессуары (ac, %)': f"{model_metrics.get('accessories', 0):.1f}",
            'Очки (gl, %)': f"{model_metrics.get('glasses', 0):.1f}",
            'Маски (ms, %)': f"{model_metrics.get('mask', 0):.1f}",
            'Плохие условия (bad, %)': f"{model_metrics.get('bad_quality', 0):.1f}",
            'Точность (challenge, %)': f"{model_metrics['challenge_total']:.1f}"
        }
        table_data.append(row)
    
    metrics_df = pd.DataFrame(table_data)
    metrics_df.to_csv(f'{report_dir}/metrics_table.csv', index=False)
    plt.style.use('default')
    
    # 1. График времени обработки
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df['Модель'], metrics_df['Время обработки (мс)'].astype(float))
    plt.title('Время обработки изображений')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{report_dir}/processing_time.png')
    plt.close()
    
    # 2. График точности для разных типов тестов
    accuracy_metrics = ['Точность (front, %)', 'Точность (pairs, %)', 
                       'Аксессуары (ac, %)', 'Очки (gl, %)', 
                       'Маски (ms, %)', 'Плохие условия (bad, %)',
                       'Точность (challenge, %)']
    
    plt.figure(figsize=(12, 8))
    metrics_df_melted = pd.melt(metrics_df, 
                               id_vars=['Модель'],
                               value_vars=accuracy_metrics,
                               var_name='Метрика',
                               value_name='Точность (%)')
    
    metrics_df_melted['Точность (%)'] = metrics_df_melted['Точность (%)'].astype(float)
    
    for model in metrics_df_melted['Модель'].unique():
        model_data = metrics_df_melted[metrics_df_melted['Модель'] == model]
        plt.plot(model_data['Метрика'], model_data['Точность (%)'], 
                marker='o', label=model)
    
    plt.title('Точность по различным метрикам')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{report_dir}/accuracy_metrics.png')
    plt.close()
    
    # 3. Создаем столбчатые диаграммы
    distance_stats = analyze_distances(results_df)
    create_bar_charts(metrics_df, distance_stats, report_dir)

    with open(f'{report_dir}/detailed_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== Детальный отчет по тестированию ===\n\n")
        
        f.write("Таблица 1. Результаты тестирования\n")
        f.write("=" * 100 + "\n")
        f.write(metrics_df.to_string(index=False) + "\n\n")
        
        f.write("=== Анализ расстояний между эмбеддингами ===\n")
        for model, stats in distance_stats.items():
            f.write(f"\n{model}:\n")
            f.write(f"  Позитивные пары: {stats['positive_mean']:.4f} ± {stats['positive_std']:.4f}\n")
            f.write(f"  Негативные пары: {stats['negative_mean']:.4f} ± {stats['negative_std']:.4f}\n")


if __name__ == '__main__':
    generate_report()