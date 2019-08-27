from collections import defaultdict
import json
import os

from circleci.api import Api


DESTDIR = os.environ.get('DESTDIR', None)
CIRCLE_TOKEN = os.environ.get('CIRCLE_TOKEN', None)
assert len(CIRCLE_TOKEN) >= 1, "Missing CIRCLE_TOKEN environment variable."


circleci = Api(CIRCLE_TOKEN)

def find_most_recent_deploy_builds(deploy_job_name_prefix='deploy-'):

  all_recent_builds = circleci.get_project_build_summary(
      username='tf-encrypted',
      project='tf-big',
      limit=50,
      status_filter=None,
      branch='master',
      vcs_type='github',
  )

  commit_filter = None
  most_recent_deploy_builds = list()

  for build in all_recent_builds:
    job_name = build['workflows']['job_name']
    commit = build['all_commit_details'][0]['commit']

    # skip all jobs we're not interested in
    if not job_name.startswith(deploy_job_name_prefix):
      continue

    # we only want the most recent builds
    if commit_filter is None:
      # we have found our commit filter (first commit encountered)
      commit_filter = commit
    else:
      # we have a commit filter, now apply it
      if commit != commit_filter:
        continue

    most_recent_deploy_builds.append(build)

  return most_recent_deploy_builds

def download_artifacts_from_builds(builds, destdir=None):
  for build in deploy_builds:
    artifacts = circleci.get_artifacts(
        username='tf-encrypted',
        project='tf-big',
        build_num=build['build_num'],
        vcs_type='github',
    )
    print("Processing build {build_num} ({build_url}) committed at {committer_date} and stopped at {stop_time}".format(
        build_num=build['build_num'],
        committer_date=build['committer_date'],
        stop_time=build['stop_time'],
        build_url=build['build_url'],
    ))
    for artifact in artifacts:
      print(" - Downloading artifact '{pretty_part}'".format(
          pretty_part=artifact['pretty_path'],
      ))
      circleci.download_artifact(
          url=artifact['url'],
          destdir=destdir,
      )

deploy_builds = find_most_recent_deploy_builds()
download_artifacts_from_builds(deploy_builds, destdir=DESTDIR)
