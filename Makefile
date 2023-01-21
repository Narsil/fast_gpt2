.PHONY: docker_anaconda docker
purge_disk:
	sync # (move data, modified through FS -> HDD cache) + flush HDD cache
	echo 3 > sudo /proc/sys/vm/drop_caches # (slab + pagecache) -> HDD (https://www.kernel.org/doc/Documentation/sysctl/vm.txt)
	sudo blockdev --flushbufs /dev/sda
	sudo hdparm -F /dev/sda

docker:
	docker build -f docker/Dockerfile . -t fast_gpt2

docker_anaconda:
	docker build -f docker/Dockerfile . -t fast_gpt2 --build-arg ANACONDA_ROOT=${ANACONDA_ROOT}

