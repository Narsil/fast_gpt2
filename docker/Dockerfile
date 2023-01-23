FROM rust:1.66 as build

RUN cd /tmp && \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt update && \
    apt install -y intel-basekit

SHELL ["/bin/bash", "-c"]
ENV STATIC=1

WORKDIR /usr/src/fast_gpt2
COPY docker/dummy.rs .
COPY Cargo.toml .
COPY Cargo.lock .
RUN sed -i 's#src/main.rs#dummy.rs#' Cargo.toml
RUN source /opt/intel/oneapi/setvars.sh && cargo build --release --features intel-mkl
RUN sed -i 's#dummy.rs#src/main.rs#' Cargo.toml
COPY src/ src/
COPY . . 

RUN source /opt/intel/oneapi/setvars.sh && cargo build --release --features intel-mkl

FROM gcr.io/distroless/cc-debian10

COPY --from=build /usr/src/fast_gpt2/target/release/fast_gpt2 /usr/local/bin/fast_gpt2
COPY --from=build /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so /usr/lib/
ENV PORT=80
# COPY tokenizer.json .
# COPY model.safetensors .
CMD ["fast_gpt2"]


