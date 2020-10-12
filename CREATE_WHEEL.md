# Steps to create a wheel and upload it.

1. Make sure you have gsutil installed or follow the install instructions in the thirdwave README.
1. Inside docker create the wheel by running:
    ```bash
    python3 setup.py sdist bdist_wheel
    ```
1. Upload the dist folder via gsutil
    ```bash
    gsutil cp -r dist gs://twa-build-support/detectron_dist/$(date +"%Y_%m_%d")
    ```

