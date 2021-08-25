# Build and Deploy Steps
Here are the details in how to Build and Deploy the changes to PyPl. Note that just because you do a new build doesn't mean there will be a deploy, however it's encouraged that that's done.

### Table of Contents
1. [Build Steps](#build-steps)
2. [Deploy Steps](#deploy-steps)


## Build Steps
To build the new docker env and prepare for a new release, follow the steps below:
1. Save and commit all of your changes.
2. Run **`scripts\predeploy.bat`**. This will do a bumpversion patch and build the environment, creating the necessary directories for testing. It will then call the **`scripts\build.bat`** script, which will rebuild the binaries for the pip package with the newest patch version.
3. Run **`scripts\docker_build.bat`**. This will build the docker container with the new tags, but it will NOT push it to the repo. That will have to be done after this entire process.
4. Run **`scripts\docker_run.bat`**. This will run a container using the image we built in the previous step. The interactive shell should start with the environment activated. If it isn't, something went wrong, likely relating back to the `Dockerfile`.
5. Within the interactive shell, run **`scripts/run_all_tests.sh`**. This will run all of the tests required to make sure the environment actually built properly. If it fails, again something went wrong in the build process and more investigation will be required, as if the tests don't run locally they're not going to pass with the git workflows either.
6. If the tests pass, then we're ready to commit everything and push to GitHub. Add an appropriate commit message and push to the repo.
7. Go to the Actions tab within the repo and check the **`build-and-test-env`** action. If this action fails, once again something is wrong with the environment and you'll have to debug and find out.
8. If the **`build-and-test-env`** action passes, open a pull request to **`develop`** or **`master`**. Either one is fine, and later **`develop`** can be merged into **`master`**, however typically merging into the **`master`** branch implies we're getting ready to release a new version.


## Deploy Steps
If you did a merge into **`master`** in the previous steps, then it's a good idea to create a new release.
1. Go to the Tags page on GitHub and click the three dots next to the newest tag. Select **Create Release**.
2. From here, you can name the release and add a brief description of what changed between this version and the previous version. Once you're done, click **release**.
3. The **Publish Package** action will kick off which will deploy the newest version to PyPl, and do the required updates. This ***SHOULD*** run properly. If it doesn't then ~~cry~~ look at the logs to try and debug the issue.