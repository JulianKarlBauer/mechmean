# New Release
Github-CI connection to PyPI is triggered by push of tagged commits.
Therefore, create a new commit,
then tag it locally

```shell
git tag -a v0.0.3 -m "new release"
```

and then push including tags

```shell
git push --tags
```

- Remove doc-strings using Find&Replace in Atom
	- Regex: r'''[\s\S]*?'''