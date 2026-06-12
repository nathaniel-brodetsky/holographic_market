# CUDA Environment Setup

Если после установки `nvcc` не виден или `nvidia-smi` не находит GPU — проблема в путях.
Это случается почти всегда при ручной установке CUDA Toolkit.

---

## Быстрая диагностика

```bash
nvcc --version          # должен показать версию
nvidia-smi              # должен показать GPU
echo $PATH | grep cuda  # должен содержать /usr/local/cuda/bin
```

Если что-то из этого пустое или "not found" — читай дальше.

---

## Шаг 1 — Экспортировать пути (сделать прямо сейчас)

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

После этого `nvcc --version` должен заработать в текущем терминале.

---

## Шаг 2 — Сделать постоянным (чтобы не вводить каждый раз)

### Fedora / любой systemd-дистрибутив

```bash
sudo tee /etc/profile.d/cuda.sh << 'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
EOF
```

Применится при следующем логине. Применить сейчас:

```bash
source /etc/profile.d/cuda.sh
```

### Ubuntu (альтернатива — через ~/.bashrc)

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
source ~/.bashrc
```

---

## Шаг 3 — Пути для cuML (RAPIDS / conda)

Если установлен RAPIDS через Miniforge:

```bash
export CUML_ROOT=/opt/miniforge3/envs/rapids
export CMAKE_PREFIX_PATH=${CUML_ROOT}:${CMAKE_PREFIX_PATH:-}
```

Сделать постоянным:

```bash
sudo tee /etc/profile.d/cuml.sh << EOF
export CUML_ROOT=/opt/miniforge3/envs/rapids
export CMAKE_PREFIX_PATH=\${CUML_ROOT}:\${CMAKE_PREFIX_PATH:-}
EOF
```

---

## Шаг 4 — Сборка проекта

```bash
cd engine/
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCUML_ROOT=/opt/miniforge3/envs/rapids   # убрать если cuML не установлен

cmake --build build --parallel $(nproc)
```

`-DCMAKE_CUDA_ARCHITECTURES=native` — автоматически определяет архитектуру твоего GPU.
Если нужно явно: RTX 4070/L4 = `89`, A100 = `80`, RTX 3090 = `86`.

### Пример для сборки без cuML (только CUDA, без DBSCAN)

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build --parallel $(nproc)
```

---

## Шаг 5 — Запуск

```bash
# Live-режим (Binance WebSocket)
./build/holographic_live

# Backtest (Phase VI — пока stub)
./build/holographic_backtest data/lob_data.csv
```

---

## Типичные ошибки

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `nvcc: command not found` | PATH не содержит `/usr/local/cuda/bin` | Шаг 1 |
| `libcuda.so: cannot open` | LD_LIBRARY_PATH не содержит `/usr/local/cuda/lib64` | Шаг 1 |
| `CUDA driver version insufficient` | Драйвер старше Toolkit | Обновить драйвер NVIDIA |
| `no kernel image available` | CUDA_ARCHITECTURES не совпадает с GPU | Указать правильный arch или `native` |
| `cuML headers not found` | CUML_ROOT не выставлен | Шаг 3 или `-DCUML_ROOT=` в cmake |
| SELinux блокирует CUDA (Fedora) | SELinux Enforcing | `sudo setenforce 0` (временно) |

---

## Проверить всё сразу

```bash
echo "=== nvcc ===" && nvcc --version
echo "=== GPU ===" && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=== PATH ===" && echo $PATH | tr ':' '\n' | grep cuda
echo "=== LD_LIBRARY_PATH ===" && echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
echo "=== cuML ===" && ls $CUML_ROOT/include/cuml 2>/dev/null | head -3 || echo "cuML не найден"
```
