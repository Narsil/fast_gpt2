FROM rust:1.66 as build

RUN cd /tmp && \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt update && \
    apt install -y intel-oneapi-openmp-2022.1.0 \
                   intel-oneapi-mkl-2022.1.0

SHELL ["/bin/bash", "-c"]

WORKDIR /usr/src/fast_gpt2
COPY docker/dummy.rs .
COPY Cargo.toml .
COPY Cargo.lock .
RUN sed -i 's#src/main.rs#dummy.rs#' Cargo.toml
RUN source /opt/intel/oneapi/setvars.sh && cargo build --release --features intel-mkl
RUN sed -i 's#dummy.rs#src/main.rs#' Cargo.toml
COPY src/ src/
COPY . . 

ARG ANACONDA_ROOT
ENV ANACONDA_ROOT=${ANACONDA_ROOT}
RUN source /opt/intel/oneapi/setvars.sh && cargo build --release --features intel-mkl

FROM gcr.io/distroless/cc-debian10

COPY --from=build /usr/src/fast_gpt2/target/release/fast_gpt2 /usr/local/bin/fast_gpt2
COPY --from=build /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so.2 /usr/lib/
COPY --from=build /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so.2 /usr/lib/
COPY --from=build /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so.2 /usr/lib/
COPY --from=build /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_def.so.2 /usr/lib/
COPY --from=build /opt/intel/oneapi/compiler/2022.1.0/linux/compiler/lib/intel64_lin/libiomp5.so /usr/lib/
# COPY tokenizer.json .
# COPY model.safetensors .
CMD ["fast_gpt2"]


