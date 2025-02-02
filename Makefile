TAG 			:= latest
USER 			:= mlexchange
PROJECT			:= dimension-reduction-pca

IMG_WEB_SVC    		:= ${USER}/${PROJECT}:${TAG}
IMG_WEB_SVC_JYP    	:= ${USER}/${PROJECT_JYP}:${TAG}
ID_USER			:= ${shell id -u}
ID_GROUP			:= ${shell id -g}

.PHONY:

test:
	echo ${IMG_WEB_SVC}
	echo ${TAG}
	echo ${PROJECT}
	echo ${PROJECT}:${TAG}
	echo ${ID_USER}

build_docker:
	docker build -t ${IMG_WEB_SVC} -f ./Dockerfile .

build_podman:
	podman build -t ghcr.io/runboj/mlex_dimension_reduction_pca:main -f ./docker/Dockerfile .

run_docker:
	docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}/data:/app/work/data/ ${IMG_WEB_SVC} bash

build_podman:
	podman build -t ghcr.io/runboj/mlex_dimension_reduction_umap:main -f ./Dockerfile .

PCA_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ ${IMG_WEB_SVC} python pca_run.py example_pca.yaml

push_docker:
	docker push ${IMG_WEB_SVC}
clean:
	find -name "*~" -delete
	-rm .python_history
	-rm -rf .config
	-rm -rf .cache
