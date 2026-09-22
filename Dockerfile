# Build and run the whole suite from nothing: compiler, CMake, the mujoco
# wheel (headers + library included), microduck-rl at the pinned commit for
# the model and the Python reference, then the C++ and Python tests.
#   docker build -t mujoco-vecenv-cpp .
#   docker run --rm mujoco-vecenv-cpp
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable
WORKDIR /work

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ARG MICRODUCK_RL_COMMIT=8a1a03849ce0c11130e9ad38dc0bd719d0bfb147
RUN git clone --quiet https://github.com/AungKaung1928/microduck-rl.git /work/microduck-rl \
    && git -C /work/microduck-rl -c advice.detachedHead=false checkout --quiet ${MICRODUCK_RL_COMMIT} \
    && cd /work/microduck-rl && ./fetch_assets.sh
ENV MICRODUCK_RL=/work/microduck-rl MICRODUCK_ASSETS=/work/microduck-rl/assets

COPY . /work/mujoco-vecenv-cpp
WORKDIR /work/mujoco-vecenv-cpp
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4

CMD ["sh", "-c", "./build/vecenv_tests && python -m pytest tests/py -q -p no:cacheprovider"]
