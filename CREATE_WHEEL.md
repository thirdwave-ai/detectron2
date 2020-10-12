# Steps to create a wheel and upload it.

1. Make sure you have gsutil installed or follow the install instructions in the thirdwave README.
1. Inside docker create the wheel by running:
    ```bash
    python3 setup.py sdist bdist_wheel
    ```
1. Outside docker upload the dist folder via gsutil:
    ```bash
    gsutil cp -r dist gs://twa-build-support/detectron_dist/$(date +"%Y_%m_%d")
    ```
1. Edit the url in the `thirdwave/requirements/bazel_requirements.in` file to point to the new path and re-generate the dependencies.
    ```bash
    /usr/local/bin/pip-compile-multi --no-upgrade --use-cache
    ```

