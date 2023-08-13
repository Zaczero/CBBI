.PHONY: update

update:
	docker push $$(docker load < $$(nix-build --no-out-link) | sed -En 's/Loaded image: (\S+)/\1/p')
