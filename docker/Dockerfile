FROM rust:1.66 as build

RUN cd /tmp && \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt update && \
    apt install -y intel-oneapi-openmp \
                   intel-oneapi-mkl
RUN apt install libatlas-base-dev -y

SHELL ["/bin/bash", "-c"]

WORKDIR /usr/src/fast_gpt2
COPY docker/dummy.rs .
COPY Cargo.toml .
COPY Cargo.lock .
ENV MKL_VERSION=2023.0.0
RUN sed -i 's#src/main.rs#dummy.rs#' Cargo.toml
RUN source /opt/intel/oneapi/setvars.sh && cargo build --release --features intel-mkl
RUN sed -i 's#dummy.rs#src/main.rs#' Cargo.toml
COPY . .

# Intel OMP specific variables
ARG OMP_NUM_THREADS=8
ENV OMP_NUM_THREADS ${OMP_NUM_THREADS}

ARG KMP_BLOCKTIME=1s
ENV KMP_BLOCKTIME ${KMP_BLOCKTIME}

# Use 1 socket, num threads defined for // sections as number of cores with one thread per core
ARG KMP_HW_SUBSET=1s,${OMP_NUM_THREADS}c,1t
ENV KMP_HW_SUBSET ${KMP_HW_SUBSET}

RUN source /opt/intel/oneapi/setvars.sh && cargo build --release --features intel-mkl
RUN ldd /usr/src/fast_gpt2/target/release/fast_gpt2 && xx

FROM gcr.io/distroless/cc-debian10

COPY --from=build /usr/src/fast_gpt2/target/release/fast_gpt2 /usr/local/bin/fast_gpt2
COPY --from=build /usr/lib/x86_64-linux-gnu/libcblas.so.3 /usr/lib/
COPY --from=build /usr/lib/x86_64-linux-gnu/libatlas.so.3 /usr/lib/
COPY tokenizer.json .
COPY model.safetensors .
CMD ["fast_gpt2"]


