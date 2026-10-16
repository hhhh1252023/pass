cd /tmp
wget "https://files.pythonhosted.org/packages/76/e1/1ab34bab811905056510915898ff167ed7944356354a23dc8f50c30977a8/sglang-0.5.21-cp312-cp312-manylinux_2_34_aarch64.whl" -O sglang-0.5.21-cp312-aarch64.whl



unzip -o sglang-0.5.21-cp312-aarch64.whl 'sglang/srt/rust_extensions/*.so' -d /tmp/ext

mkdir -p /root/.cache/.cache/npu-pkgs/rust
cp /tmp/ext/sglang/srt/rust_extensions/*.so /root/.cache/.cache/npu-pkgs/rust/
ls -l /root/.cache/.cache/npu-pkgs/rust/
