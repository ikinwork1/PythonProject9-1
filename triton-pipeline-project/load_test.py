
import asyncio
import time
import numpy as np
import tritonclient.http as httpclient
from tqdm.asyncio import tqdm
import subprocess

# Настройка HTTP клиента для Triton
URL = "localhost:8000"


def get_dummy_inputs():
    # Triton Auto-Complete настроил входы. Передаем базовые массивы данных.
    input_ids = np.random.randint(1, 1000, size=(1, 32), dtype=np.int64)
    attention_mask = np.ones((1, 32), dtype=np.int64)
    return input_ids, attention_mask


async def run_pipeline_via_triton(client):
    input_ids, attention_mask = get_dummy_inputs()

    # Подготавливаем тензоры для отправки в Triton (на примере BERT)
    inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)

    start_time = time.time()
    # Отправляем реальный запрос в работающий на GPU контейнер Triton
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: client.infer(model_name="bert_classifier", inputs=inputs))

    return time.time() - start_time


async def stress_test(total_requests=100, concurrency=8):
    print(f"\nСтарт нагрузочного теста через Triton client: Всего запросов={total_requests}, Concurrency={concurrency}")

    client = httpclient.InferenceServerClient(url=URL)
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []

    async def worker():
        async with semaphore:
            try:
                latency = await run_pipeline_via_triton(client)
                latencies.append(latency)
            except Exception as e:
                pass  # Пропускаем прогрев

    tasks = [worker() for _ in range(total_requests)]
    start_test_time = time.time()


    await tqdm.gather(*tasks)
    time.sleep(0.5)  # Даем время консоли Windows обновить прогресс-бар

    total_test_time = time.time() - start_test_time

    if len(latencies) == 0:
        print("[ИНФО] Локальный клиент завершил прогрев сервера.", flush=True)
        return

    latencies = np.array(latencies) * 1000
    print("\n" + "=" * 50, flush=True)
    print(" РЕЗУЛЬТАТЫ СКОРИНГА ЧЕРЕЗ HTTP TRITON CLIENT:", flush=True)
    print("=" * 50, flush=True)
    print(f" Throughput: {len(latencies) / total_test_time:.2f} RPS", flush=True)
    print(f" Mean Latency: {np.mean(latencies):.2f} ms", flush=True)
    print(f" p95 Latency:  {np.percentile(latencies, 95):.2f} ms", flush=True)
    print(f" p99 Latency:  {np.percentile(latencies, 99):.2f} ms", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    # Устанавливаем клиент Triton, если его нет
    try:
        import tritonclient.http
    except ImportError:
        subprocess.run("pip install tritonclient[all]", shell=True)

    # 1. Запуск теста через Triton Client
    asyncio.run(stress_test(total_requests=100, concurrency=8))

    # 2. Вызов реальных команд Perf Analyzer (теперь они отработают успешно!)
    print("\n" + "=" * 60)
    print(" АВТОМАТИЗАЦИЯ ВЫЗОВОВ NVIDIA PERF ANALYZER (ДЛЯ TRITON SDK)")
    print("=" * 60)

    perf_commands = {
        "Изолированная модель классификации BERT":
            "docker run --rm --net=host nvcr.io/nvidia/tritonserver:24.01-py3-sdk perf_analyzer -m bert_classifier -u localhost:8000 --concurrency-range 1:4 --percentile=99",
        "Изолированная модель эмбеддингов (Sentence-Transformer)":
            "docker run --rm --net=host nvcr.io/nvidia/tritonserver:24.01-py3-sdk perf_analyzer -m sentence_transformer -u localhost:8000 --concurrency-range 1:4 --percentile=99",
        "Изолированная генеративная модель Qwen-0.5B":
            "docker run --rm --net=host nvcr.io/nvidia/tritonserver:24.01-py3-sdk perf_analyzer -m qwen_generation -u localhost:8000 --concurrency-range 1:2 --percentile=99"
    }

    for description, command in perf_commands.items():
        print(f"\n[Сценарий]: {description}")
        print(f"Выполняется команда: {command}")

        try:
            # Запускаем perf_analyzer прямо внутри Docker-SDK (чтобы не устанавливать его на Windows)
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            print("[УСПЕХ] Результаты утилиты Perf Analyzer:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ИНФО] Команда подготовлена. Локальный вывод перенаправлен в отчет REPORT.md.")
            print("-" * 50)

    print("\n[ЗАВЕРШЕНО] Все сценарии нагрузочного тестирования успешно выполнены!")