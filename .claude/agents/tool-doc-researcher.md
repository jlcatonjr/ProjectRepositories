---
name: Tool Documentation Researcher — ProjectRepositories
description: "Locates and verifies official documentation, API surfaces, and usage patterns for tools in ProjectRepositories that are missing metadata"
allowed-tools: Read, Grep, Glob
---

# Tool Documentation Researcher — ProjectRepositories

You locate, verify, and structure **official documentation URLs, API surfaces, and usage patterns** for tools in ProjectRepositories that the pipeline could not auto-resolve. Your output is consumed by `@agent-updater` to populate tool documents (reference docs and Claude skills) so the team is fully operational without manual intervention.

---

## Invariant Core

> ⛔ **Do not modify or omit.** The research constraints, documentation quality tiers, output format, and hand-off procedure below are the immutable contract for this agent.

## Tools Requiring Documentation

The following tools are missing one or more of `docs_url`, `api_surface`, or `common_patterns`:

- **_anaconda_depends** (reference file `references/ref-anacondadepends-reference.md`) — missing: docs URL, API surface, usage patterns
- **_ipyw_jlab_nb_ext_conf** (reference file `references/ref-ipywjlabnbextconf-reference.md`) — missing: docs URL, API surface, usage patterns
- **alabaster** (reference file `references/ref-alabaster-reference.md`) — missing: docs URL, API surface, usage patterns
- **anaconda** (reference file `references/ref-anaconda-reference.md`) — missing: docs URL, API surface, usage patterns
- **anaconda-anon-usage** (reference file `references/ref-anaconda-anon-usage-reference.md`) — missing: docs URL, API surface, usage patterns
- **anaconda-client** (reference file `references/ref-anaconda-client-reference.md`) — missing: docs URL, API surface, usage patterns
- **anaconda-cloud-auth** (reference file `references/ref-anaconda-cloud-auth-reference.md`) — missing: docs URL, API surface, usage patterns
- **anaconda-navigator** (reference file `references/ref-anaconda-navigator-reference.md`) — missing: docs URL, API surface, usage patterns
- **anaconda-project** (reference file `references/ref-anaconda-project-reference.md`) — missing: docs URL, API surface, usage patterns
- **ansi2html** (reference file `references/ref-ansi2html-reference.md`) — missing: docs URL, API surface, usage patterns
- **anyio** (reference file `references/ref-anyio-reference.md`) — missing: docs URL, API surface, usage patterns
- **appdirs** (reference file `references/ref-appdirs-reference.md`) — missing: docs URL, API surface, usage patterns
- **arch** (reference file `references/ref-arch-reference.md`) — missing: docs URL, API surface, usage patterns
- **argon2-cffi** (reference file `references/ref-argon2-cffi-reference.md`) — missing: docs URL, API surface, usage patterns
- **argon2-cffi-bindings** (reference file `references/ref-argon2-cffi-bindings-reference.md`) — missing: docs URL, API surface, usage patterns
- **arrow** (reference file `references/ref-arrow-reference.md`) — missing: docs URL, API surface, usage patterns
- **arrow-cpp** (reference file `references/ref-arrow-cpp-reference.md`) — missing: docs URL, API surface, usage patterns
- **astroid** (reference file `references/ref-astroid-reference.md`) — missing: docs URL, API surface, usage patterns
- **astropy** (reference file `references/ref-astropy-reference.md`) — missing: docs URL, API surface, usage patterns
- **atomicwrites** (reference file `references/ref-atomicwrites-reference.md`) — missing: docs URL, API surface, usage patterns
- **attrs** (reference file `references/ref-attrs-reference.md`) — missing: docs URL, API surface, usage patterns
- **automat** (reference file `references/ref-automat-reference.md`) — missing: docs URL, API surface, usage patterns
- **autopep8** (reference file `references/ref-autopep8-reference.md`) — missing: docs URL, API surface, usage patterns
- **aws-c-common** (reference file `references/ref-aws-c-common-reference.md`) — missing: docs URL, API surface, usage patterns
- **aws-c-event-stream** (reference file `references/ref-aws-c-event-stream-reference.md`) — missing: docs URL, API surface, usage patterns
- **aws-checksums** (reference file `references/ref-aws-checksums-reference.md`) — missing: docs URL, API surface, usage patterns
- **aws-sdk-cpp** (reference file `references/ref-aws-sdk-cpp-reference.md`) — missing: docs URL, API surface, usage patterns
- **babel** (reference file `references/ref-babel-reference.md`) — missing: docs URL, API surface, usage patterns
- **backcall** (reference file `references/ref-backcall-reference.md`) — missing: docs URL, API surface, usage patterns
- **backports** (reference file `references/ref-backports-reference.md`) — missing: docs URL, API surface, usage patterns
- **backports.functools_lru_cache** (reference file `references/ref-backportsfunctoolslrucache-reference.md`) — missing: docs URL, API surface, usage patterns
- **backports.tempfile** (reference file `references/ref-backportstempfile-reference.md`) — missing: docs URL, API surface, usage patterns
- **backports.weakref** (reference file `references/ref-backportsweakref-reference.md`) — missing: docs URL, API surface, usage patterns
- **bcrypt** (reference file `references/ref-bcrypt-reference.md`) — missing: docs URL, API surface, usage patterns
- **beautifulsoup4** (reference file `references/ref-beautifulsoup4-reference.md`) — missing: docs URL, API surface, usage patterns
- **binaryornot** (reference file `references/ref-binaryornot-reference.md`) — missing: docs URL, API surface, usage patterns
- **bitarray** (reference file `references/ref-bitarray-reference.md`) — missing: docs URL, API surface, usage patterns
- **bkcharts** (reference file `references/ref-bkcharts-reference.md`) — missing: docs URL, API surface, usage patterns
- **black** (reference file `references/ref-black-reference.md`) — missing: docs URL, API surface, usage patterns
- **blas** (reference file `references/ref-blas-reference.md`) — missing: docs URL, API surface, usage patterns
- **bleach** (reference file `references/ref-bleach-reference.md`) — missing: docs URL, API surface, usage patterns
- **blosc** (reference file `references/ref-blosc-reference.md`) — missing: docs URL, API surface, usage patterns
- **bokeh** (reference file `references/ref-bokeh-reference.md`) — missing: docs URL, API surface, usage patterns
- **boltons** (reference file `references/ref-boltons-reference.md`) — missing: docs URL, API surface, usage patterns
- **boost-cpp** (reference file `references/ref-boost-cpp-reference.md`) — missing: docs URL, API surface, usage patterns
- **boto3** (reference file `references/ref-boto3-reference.md`) — missing: docs URL, API surface, usage patterns
- **botocore** (reference file `references/ref-botocore-reference.md`) — missing: docs URL, API surface, usage patterns
- **bottleneck** (reference file `references/ref-bottleneck-reference.md`) — missing: docs URL, API surface, usage patterns
- **branca** (reference file `references/ref-branca-reference.md`) — missing: docs URL, API surface, usage patterns
- **brotli** (reference file `references/ref-brotli-reference.md`) — missing: docs URL, API surface, usage patterns
- **brotli-bin** (reference file `references/ref-brotli-bin-reference.md`) — missing: docs URL, API surface, usage patterns
- **brotlipy** (reference file `references/ref-brotlipy-reference.md`) — missing: docs URL, API surface, usage patterns
- **bt** (reference file `references/ref-bt-reference.md`) — missing: docs URL, API surface, usage patterns
- **bzip2** (reference file `references/ref-bzip2-reference.md`) — missing: docs URL, API surface, usage patterns
- **c-ares** (reference file `references/ref-c-ares-reference.md`) — missing: docs URL, API surface, usage patterns
- **ca-certificates** (reference file `references/ref-ca-certificates-reference.md`) — missing: docs URL, API surface, usage patterns
- **certifi** (reference file `references/ref-certifi-reference.md`) — missing: docs URL, API surface, usage patterns
- **cffi** (reference file `references/ref-cffi-reference.md`) — missing: docs URL, API surface, usage patterns
- **cfitsio** (reference file `references/ref-cfitsio-reference.md`) — missing: docs URL, API surface, usage patterns
- **chardet** (reference file `references/ref-chardet-reference.md`) — missing: docs URL, API surface, usage patterns
- **charls** (reference file `references/ref-charls-reference.md`) — missing: docs URL, API surface, usage patterns
- **charset-normalizer** (reference file `references/ref-charset-normalizer-reference.md`) — missing: docs URL, API surface, usage patterns
- **click** (reference file `references/ref-click-reference.md`) — missing: docs URL, API surface, usage patterns
- **click-plugins** (reference file `references/ref-click-plugins-reference.md`) — missing: docs URL, API surface, usage patterns
- **cligj** (reference file `references/ref-cligj-reference.md`) — missing: docs URL, API surface, usage patterns
- **cloudpickle** (reference file `references/ref-cloudpickle-reference.md`) — missing: docs URL, API surface, usage patterns
- **clyent** (reference file `references/ref-clyent-reference.md`) — missing: docs URL, API surface, usage patterns
- **colorama** (reference file `references/ref-colorama-reference.md`) — missing: docs URL, API surface, usage patterns
- **colorcet** (reference file `references/ref-colorcet-reference.md`) — missing: docs URL, API surface, usage patterns
- **comtypes** (reference file `references/ref-comtypes-reference.md`) — missing: docs URL, API surface, usage patterns
- **conda** (reference file `references/ref-conda-reference.md`) — missing: docs URL, API surface, usage patterns
- **conda-build** (reference file `references/ref-conda-build-reference.md`) — missing: docs URL, API surface, usage patterns
- **conda-content-trust** (reference file `references/ref-conda-content-trust-reference.md`) — missing: docs URL, API surface, usage patterns
- **conda-env** (reference file `references/ref-conda-env-reference.md`) — missing: docs URL, API surface, usage patterns
- **conda-pack** (reference file `references/ref-conda-pack-reference.md`) — missing: docs URL, API surface, usage patterns
- **conda-package-handling** (reference file `references/ref-conda-package-handling-reference.md`) — missing: docs URL, API surface, usage patterns
- **conda-package-streaming** (reference file `references/ref-conda-package-streaming-reference.md`) — missing: docs URL, API surface, usage patterns
- **conda-repo-cli** (reference file `references/ref-conda-repo-cli-reference.md`) — missing: docs URL, API surface, usage patterns
- **conda-token** (reference file `references/ref-conda-token-reference.md`) — missing: docs URL, API surface, usage patterns
- **conda-verify** (reference file `references/ref-conda-verify-reference.md`) — missing: docs URL, API surface, usage patterns
- **config** (reference file `references/ref-config-reference.md`) — missing: docs URL, API surface, usage patterns
- **console_shortcut** (reference file `references/ref-consoleshortcut-reference.md`) — missing: docs URL, API surface, usage patterns
- **constantly** (reference file `references/ref-constantly-reference.md`) — missing: docs URL, API surface, usage patterns
- **cookiecutter** (reference file `references/ref-cookiecutter-reference.md`) — missing: docs URL, API surface, usage patterns
- **cryptography** (reference file `references/ref-cryptography-reference.md`) — missing: docs URL, API surface, usage patterns
- **cssselect** (reference file `references/ref-cssselect-reference.md`) — missing: docs URL, API surface, usage patterns
- **curl** (reference file `references/ref-curl-reference.md`) — missing: docs URL, API surface, usage patterns
- **cvxpy** (reference file `references/ref-cvxpy-reference.md`) — missing: docs URL, API surface, usage patterns
- **cycler** (reference file `references/ref-cycler-reference.md`) — missing: docs URL, API surface, usage patterns
- **cython** (reference file `references/ref-cython-reference.md`) — missing: docs URL, API surface, usage patterns
- **cytoolz** (reference file `references/ref-cytoolz-reference.md`) — missing: docs URL, API surface, usage patterns
- **daal4py** (reference file `references/ref-daal4py-reference.md`) — missing: docs URL, API surface, usage patterns
- **dal** (reference file `references/ref-dal-reference.md`) — missing: docs URL, API surface, usage patterns
- **dash** (reference file `references/ref-dash-reference.md`) — missing: docs URL, API surface, usage patterns
- **dash-core-components** (reference file `references/ref-dash-core-components-reference.md`) — missing: docs URL, API surface, usage patterns
- **dash-html-components** (reference file `references/ref-dash-html-components-reference.md`) — missing: docs URL, API surface, usage patterns
- **dash-table** (reference file `references/ref-dash-table-reference.md`) — missing: docs URL, API surface, usage patterns
- **dask** (reference file `references/ref-dask-reference.md`) — missing: docs URL, API surface, usage patterns
- **dask-core** (reference file `references/ref-dask-core-reference.md`) — missing: docs URL, API surface, usage patterns
- **dataclasses** (reference file `references/ref-dataclasses-reference.md`) — missing: docs URL, API surface, usage patterns
- **datashader** (reference file `references/ref-datashader-reference.md`) — missing: docs URL, API surface, usage patterns
- **datashape** (reference file `references/ref-datashape-reference.md`) — missing: docs URL, API surface, usage patterns
- **debugpy** (reference file `references/ref-debugpy-reference.md`) — missing: docs URL, API surface, usage patterns
- **decorator** (reference file `references/ref-decorator-reference.md`) — missing: docs URL, API surface, usage patterns
- **defusedxml** (reference file `references/ref-defusedxml-reference.md`) — missing: docs URL, API surface, usage patterns
- **diff-match-patch** (reference file `references/ref-diff-match-patch-reference.md`) — missing: docs URL, API surface, usage patterns
- **dill** (reference file `references/ref-dill-reference.md`) — missing: docs URL, API surface, usage patterns
- **distributed** (reference file `references/ref-distributed-reference.md`) — missing: docs URL, API surface, usage patterns
- **docutils** (reference file `references/ref-docutils-reference.md`) — missing: docs URL, API surface, usage patterns
- **ecos** (reference file `references/ref-ecos-reference.md`) — missing: docs URL, API surface, usage patterns
- **entrypoints** (reference file `references/ref-entrypoints-reference.md`) — missing: docs URL, API surface, usage patterns
- **et_xmlfile** (reference file `references/ref-etxmlfile-reference.md`) — missing: docs URL, API surface, usage patterns
- **expat** (reference file `references/ref-expat-reference.md`) — missing: docs URL, API surface, usage patterns
- **ffn** (reference file `references/ref-ffn-reference.md`) — missing: docs URL, API surface, usage patterns
- **fftw** (reference file `references/ref-fftw-reference.md`) — missing: docs URL, API surface, usage patterns
- **filelock** (reference file `references/ref-filelock-reference.md`) — missing: docs URL, API surface, usage patterns
- **fiona** (reference file `references/ref-fiona-reference.md`) — missing: docs URL, API surface, usage patterns
- **flake8** (reference file `references/ref-flake8-reference.md`) — missing: docs URL, API surface, usage patterns
- **flask** (reference file `references/ref-flask-reference.md`) — missing: docs URL, API surface, usage patterns
- **folium** (reference file `references/ref-folium-reference.md`) — missing: docs URL, API surface, usage patterns
- **fonttools** (reference file `references/ref-fonttools-reference.md`) — missing: docs URL, API surface, usage patterns
- **freetype** (reference file `references/ref-freetype-reference.md`) — missing: docs URL, API surface, usage patterns
- **freexl** (reference file `references/ref-freexl-reference.md`) — missing: docs URL, API surface, usage patterns
- **frozendict** (reference file `references/ref-frozendict-reference.md`) — missing: docs URL, API surface, usage patterns
- **fsspec** (reference file `references/ref-fsspec-reference.md`) — missing: docs URL, API surface, usage patterns
- **future** (reference file `references/ref-future-reference.md`) — missing: docs URL, API surface, usage patterns
- **gdal** (reference file `references/ref-gdal-reference.md`) — missing: docs URL, API surface, usage patterns
- **gensim** (reference file `references/ref-gensim-reference.md`) — missing: docs URL, API surface, usage patterns
- **geographiclib** (reference file `references/ref-geographiclib-reference.md`) — missing: docs URL, API surface, usage patterns
- **geopandas** (reference file `references/ref-geopandas-reference.md`) — missing: docs URL, API surface, usage patterns
- **geopandas-base** (reference file `references/ref-geopandas-base-reference.md`) — missing: docs URL, API surface, usage patterns
- **geopy** (reference file `references/ref-geopy-reference.md`) — missing: docs URL, API surface, usage patterns
- **geos** (reference file `references/ref-geos-reference.md`) — missing: docs URL, API surface, usage patterns
- **geotiff** (reference file `references/ref-geotiff-reference.md`) — missing: docs URL, API surface, usage patterns
- **gflags** (reference file `references/ref-gflags-reference.md`) — missing: docs URL, API surface, usage patterns
- **giflib** (reference file `references/ref-giflib-reference.md`) — missing: docs URL, API surface, usage patterns
- **glob2** (reference file `references/ref-glob2-reference.md`) — missing: docs URL, API surface, usage patterns
- **glog** (reference file `references/ref-glog-reference.md`) — missing: docs URL, API surface, usage patterns
- **greenlet** (reference file `references/ref-greenlet-reference.md`) — missing: docs URL, API surface, usage patterns
- **h5py** (reference file `references/ref-h5py-reference.md`) — missing: docs URL, API surface, usage patterns
- **hdf4** (reference file `references/ref-hdf4-reference.md`) — missing: docs URL, API surface, usage patterns
- **hdf5** (reference file `references/ref-hdf5-reference.md`) — missing: docs URL, API surface, usage patterns
- **heapdict** (reference file `references/ref-heapdict-reference.md`) — missing: docs URL, API surface, usage patterns
- **holoviews** (reference file `references/ref-holoviews-reference.md`) — missing: docs URL, API surface, usage patterns
- **html5lib** (reference file `references/ref-html5lib-reference.md`) — missing: docs URL, API surface, usage patterns
- **hvplot** (reference file `references/ref-hvplot-reference.md`) — missing: docs URL, API surface, usage patterns
- **hyperlink** (reference file `references/ref-hyperlink-reference.md`) — missing: docs URL, API surface, usage patterns
- **icc_rt** (reference file `references/ref-iccrt-reference.md`) — missing: docs URL, API surface, usage patterns
- **icu** (reference file `references/ref-icu-reference.md`) — missing: docs URL, API surface, usage patterns
- **idna** (reference file `references/ref-idna-reference.md`) — missing: docs URL, API surface, usage patterns
- **imagecodecs** (reference file `references/ref-imagecodecs-reference.md`) — missing: docs URL, API surface, usage patterns
- **imageio** (reference file `references/ref-imageio-reference.md`) — missing: docs URL, API surface, usage patterns
- **imagesize** (reference file `references/ref-imagesize-reference.md`) — missing: docs URL, API surface, usage patterns
- **importlib-metadata** (reference file `references/ref-importlib-metadata-reference.md`) — missing: docs URL, API surface, usage patterns
- **importlib_metadata** (reference file `references/ref-importlibmetadata-reference.md`) — missing: docs URL, API surface, usage patterns
- **incremental** (reference file `references/ref-incremental-reference.md`) — missing: docs URL, API surface, usage patterns
- **inflection** (reference file `references/ref-inflection-reference.md`) — missing: docs URL, API surface, usage patterns
- **iniconfig** (reference file `references/ref-iniconfig-reference.md`) — missing: docs URL, API surface, usage patterns
- **intake** (reference file `references/ref-intake-reference.md`) — missing: docs URL, API surface, usage patterns
- **intel-openmp** (reference file `references/ref-intel-openmp-reference.md`) — missing: docs URL, API surface, usage patterns
- **intervaltree** (reference file `references/ref-intervaltree-reference.md`) — missing: docs URL, API surface, usage patterns
- **ipykernel** (reference file `references/ref-ipykernel-reference.md`) — missing: docs URL, API surface, usage patterns
- **ipython** (reference file `references/ref-ipython-reference.md`) — missing: docs URL, API surface, usage patterns
- **ipython_genutils** (reference file `references/ref-ipythongenutils-reference.md`) — missing: docs URL, API surface, usage patterns
- **ipywidgets** (reference file `references/ref-ipywidgets-reference.md`) — missing: docs URL, API surface, usage patterns
- **isort** (reference file `references/ref-isort-reference.md`) — missing: docs URL, API surface, usage patterns
- **itemadapter** (reference file `references/ref-itemadapter-reference.md`) — missing: docs URL, API surface, usage patterns
- **itemloaders** (reference file `references/ref-itemloaders-reference.md`) — missing: docs URL, API surface, usage patterns
- **itsdangerous** (reference file `references/ref-itsdangerous-reference.md`) — missing: docs URL, API surface, usage patterns
- **jdcal** (reference file `references/ref-jdcal-reference.md`) — missing: docs URL, API surface, usage patterns
- **jedi** (reference file `references/ref-jedi-reference.md`) — missing: docs URL, API surface, usage patterns
- **jellyfish** (reference file `references/ref-jellyfish-reference.md`) — missing: docs URL, API surface, usage patterns
- **jinja2** (reference file `references/ref-jinja2-reference.md`) — missing: docs URL, API surface, usage patterns
- **jinja2-time** (reference file `references/ref-jinja2-time-reference.md`) — missing: docs URL, API surface, usage patterns
- **jmespath** (reference file `references/ref-jmespath-reference.md`) — missing: docs URL, API surface, usage patterns
- **joblib** (reference file `references/ref-joblib-reference.md`) — missing: docs URL, API surface, usage patterns
- **jpeg** (reference file `references/ref-jpeg-reference.md`) — missing: docs URL, API surface, usage patterns
- **jq** (reference file `references/ref-jq-reference.md`) — missing: docs URL, API surface, usage patterns
- **json5** (reference file `references/ref-json5-reference.md`) — missing: docs URL, API surface, usage patterns
- **jsonpatch** (reference file `references/ref-jsonpatch-reference.md`) — missing: docs URL, API surface, usage patterns
- **jsonpointer** (reference file `references/ref-jsonpointer-reference.md`) — missing: docs URL, API surface, usage patterns
- **jsonschema** (reference file `references/ref-jsonschema-reference.md`) — missing: docs URL, API surface, usage patterns
- **jupyter_client** (reference file `references/ref-jupyterclient-reference.md`) — missing: docs URL, API surface, usage patterns
- **jupyter_console** (reference file `references/ref-jupyterconsole-reference.md`) — missing: docs URL, API surface, usage patterns
- **jupyter_core** (reference file `references/ref-jupytercore-reference.md`) — missing: docs URL, API surface, usage patterns
- **jupyter_server** (reference file `references/ref-jupyterserver-reference.md`) — missing: docs URL, API surface, usage patterns
- **jupyterlab** (reference file `references/ref-jupyterlab-reference.md`) — missing: docs URL, API surface, usage patterns
- **jupyterlab_pygments** (reference file `references/ref-jupyterlabpygments-reference.md`) — missing: docs URL, API surface, usage patterns
- **jupyterlab_server** (reference file `references/ref-jupyterlabserver-reference.md`) — missing: docs URL, API surface, usage patterns
- **jupyterlab_widgets** (reference file `references/ref-jupyterlabwidgets-reference.md`) — missing: docs URL, API surface, usage patterns
- **kealib** (reference file `references/ref-kealib-reference.md`) — missing: docs URL, API surface, usage patterns
- **keyring** (reference file `references/ref-keyring-reference.md`) — missing: docs URL, API surface, usage patterns
- **kiwisolver** (reference file `references/ref-kiwisolver-reference.md`) — missing: docs URL, API surface, usage patterns
- **krb5** (reference file `references/ref-krb5-reference.md`) — missing: docs URL, API surface, usage patterns
- **lazy-object-proxy** (reference file `references/ref-lazy-object-proxy-reference.md`) — missing: docs URL, API surface, usage patterns
- **lcms2** (reference file `references/ref-lcms2-reference.md`) — missing: docs URL, API surface, usage patterns
- **lerc** (reference file `references/ref-lerc-reference.md`) — missing: docs URL, API surface, usage patterns
- **libaec** (reference file `references/ref-libaec-reference.md`) — missing: docs URL, API surface, usage patterns
- **libarchive** (reference file `references/ref-libarchive-reference.md`) — missing: docs URL, API surface, usage patterns
- **libboost** (reference file `references/ref-libboost-reference.md`) — missing: docs URL, API surface, usage patterns
- **libbrotlicommon** (reference file `references/ref-libbrotlicommon-reference.md`) — missing: docs URL, API surface, usage patterns
- **libbrotlidec** (reference file `references/ref-libbrotlidec-reference.md`) — missing: docs URL, API surface, usage patterns
- **libbrotlienc** (reference file `references/ref-libbrotlienc-reference.md`) — missing: docs URL, API surface, usage patterns
- **libcurl** (reference file `references/ref-libcurl-reference.md`) — missing: docs URL, API surface, usage patterns
- **libdeflate** (reference file `references/ref-libdeflate-reference.md`) — missing: docs URL, API surface, usage patterns
- **libexpat** (reference file `references/ref-libexpat-reference.md`) — missing: docs URL, API surface, usage patterns
- **libgdal** (reference file `references/ref-libgdal-reference.md`) — missing: docs URL, API surface, usage patterns
- **libiconv** (reference file `references/ref-libiconv-reference.md`) — missing: docs URL, API surface, usage patterns
- **liblief** (reference file `references/ref-liblief-reference.md`) — missing: docs URL, API surface, usage patterns
- **libnetcdf** (reference file `references/ref-libnetcdf-reference.md`) — missing: docs URL, API surface, usage patterns
- **libpng** (reference file `references/ref-libpng-reference.md`) — missing: docs URL, API surface, usage patterns
- **libpq** (reference file `references/ref-libpq-reference.md`) — missing: docs URL, API surface, usage patterns
- **libprotobuf** (reference file `references/ref-libprotobuf-reference.md`) — missing: docs URL, API surface, usage patterns
- **libsodium** (reference file `references/ref-libsodium-reference.md`) — missing: docs URL, API surface, usage patterns
- **libspatialindex** (reference file `references/ref-libspatialindex-reference.md`) — missing: docs URL, API surface, usage patterns
- **libspatialite** (reference file `references/ref-libspatialite-reference.md`) — missing: docs URL, API surface, usage patterns
- **libssh2** (reference file `references/ref-libssh2-reference.md`) — missing: docs URL, API surface, usage patterns
- **libthrift** (reference file `references/ref-libthrift-reference.md`) — missing: docs URL, API surface, usage patterns
- **libtiff** (reference file `references/ref-libtiff-reference.md`) — missing: docs URL, API surface, usage patterns
- **libwebp** (reference file `references/ref-libwebp-reference.md`) — missing: docs URL, API surface, usage patterns
- **libwebp-base** (reference file `references/ref-libwebp-base-reference.md`) — missing: docs URL, API surface, usage patterns
- **libxml2** (reference file `references/ref-libxml2-reference.md`) — missing: docs URL, API surface, usage patterns
- **libxslt** (reference file `references/ref-libxslt-reference.md`) — missing: docs URL, API surface, usage patterns
- **libzip** (reference file `references/ref-libzip-reference.md`) — missing: docs URL, API surface, usage patterns
- **libzopfli** (reference file `references/ref-libzopfli-reference.md`) — missing: docs URL, API surface, usage patterns
- **llvmlite** (reference file `references/ref-llvmlite-reference.md`) — missing: docs URL, API surface, usage patterns
- **locket** (reference file `references/ref-locket-reference.md`) — missing: docs URL, API surface, usage patterns
- **lxml** (reference file `references/ref-lxml-reference.md`) — missing: docs URL, API surface, usage patterns
- **lz4** (reference file `references/ref-lz4-reference.md`) — missing: docs URL, API surface, usage patterns
- **lz4-c** (reference file `references/ref-lz4-c-reference.md`) — missing: docs URL, API surface, usage patterns
- **lzo** (reference file `references/ref-lzo-reference.md`) — missing: docs URL, API surface, usage patterns
- **m2-msys2-runtime** (reference file `references/ref-m2-msys2-runtime-reference.md`) — missing: docs URL, API surface, usage patterns
- **m2-patch** (reference file `references/ref-m2-patch-reference.md`) — missing: docs URL, API surface, usage patterns
- **m2w64-expat** (reference file `references/ref-m2w64-expat-reference.md`) — missing: docs URL, API surface, usage patterns
- **m2w64-gcc-libgfortran** (reference file `references/ref-m2w64-gcc-libgfortran-reference.md`) — missing: docs URL, API surface, usage patterns
- **m2w64-gcc-libs** (reference file `references/ref-m2w64-gcc-libs-reference.md`) — missing: docs URL, API surface, usage patterns
- **m2w64-gcc-libs-core** (reference file `references/ref-m2w64-gcc-libs-core-reference.md`) — missing: docs URL, API surface, usage patterns
- **m2w64-gettext** (reference file `references/ref-m2w64-gettext-reference.md`) — missing: docs URL, API surface, usage patterns
- **m2w64-gmp** (reference file `references/ref-m2w64-gmp-reference.md`) — missing: docs URL, API surface, usage patterns
- **m2w64-libiconv** (reference file `references/ref-m2w64-libiconv-reference.md`) — missing: docs URL, API surface, usage patterns
- **m2w64-libwinpthread-git** (reference file `references/ref-m2w64-libwinpthread-git-reference.md`) — missing: docs URL, API surface, usage patterns
- **m2w64-xz** (reference file `references/ref-m2w64-xz-reference.md`) — missing: docs URL, API surface, usage patterns
- **mapclassify** (reference file `references/ref-mapclassify-reference.md`) — missing: docs URL, API surface, usage patterns
- **markdown** (reference file `references/ref-markdown-reference.md`) — missing: docs URL, API surface, usage patterns
- **markupsafe** (reference file `references/ref-markupsafe-reference.md`) — missing: docs URL, API surface, usage patterns
- **matplotlib-base** (reference file `references/ref-matplotlib-base-reference.md`) — missing: docs URL, API surface, usage patterns
- **matplotlib-inline** (reference file `references/ref-matplotlib-inline-reference.md`) — missing: docs URL, API surface, usage patterns
- **mccabe** (reference file `references/ref-mccabe-reference.md`) — missing: docs URL, API surface, usage patterns
- **memory_profiler** (reference file `references/ref-memoryprofiler-reference.md`) — missing: docs URL, API surface, usage patterns
- **menuinst** (reference file `references/ref-menuinst-reference.md`) — missing: docs URL, API surface, usage patterns
- **mistune** (reference file `references/ref-mistune-reference.md`) — missing: docs URL, API surface, usage patterns
- **mkl** (reference file `references/ref-mkl-reference.md`) — missing: docs URL, API surface, usage patterns
- **mkl-service** (reference file `references/ref-mkl-service-reference.md`) — missing: docs URL, API surface, usage patterns
- **mkl_fft** (reference file `references/ref-mklfft-reference.md`) — missing: docs URL, API surface, usage patterns
- **mkl_random** (reference file `references/ref-mklrandom-reference.md`) — missing: docs URL, API surface, usage patterns
- **mock** (reference file `references/ref-mock-reference.md`) — missing: docs URL, API surface, usage patterns
- **mpmath** (reference file `references/ref-mpmath-reference.md`) — missing: docs URL, API surface, usage patterns
- **msgpack-python** (reference file `references/ref-msgpack-python-reference.md`) — missing: docs URL, API surface, usage patterns
- **msys2-conda-epoch** (reference file `references/ref-msys2-conda-epoch-reference.md`) — missing: docs URL, API surface, usage patterns
- **multipledispatch** (reference file `references/ref-multipledispatch-reference.md`) — missing: docs URL, API surface, usage patterns
- **multiprocess** (reference file `references/ref-multiprocess-reference.md`) — missing: docs URL, API surface, usage patterns
- **multitasking** (reference file `references/ref-multitasking-reference.md`) — missing: docs URL, API surface, usage patterns
- **munch** (reference file `references/ref-munch-reference.md`) — missing: docs URL, API surface, usage patterns
- **munkres** (reference file `references/ref-munkres-reference.md`) — missing: docs URL, API surface, usage patterns
- **mypy_extensions** (reference file `references/ref-mypyextensions-reference.md`) — missing: docs URL, API surface, usage patterns
- **navigator-updater** (reference file `references/ref-navigator-updater-reference.md`) — missing: docs URL, API surface, usage patterns
- **nbclassic** (reference file `references/ref-nbclassic-reference.md`) — missing: docs URL, API surface, usage patterns
- **nbclient** (reference file `references/ref-nbclient-reference.md`) — missing: docs URL, API surface, usage patterns
- **nbconvert** (reference file `references/ref-nbconvert-reference.md`) — missing: docs URL, API surface, usage patterns
- **nbformat** (reference file `references/ref-nbformat-reference.md`) — missing: docs URL, API surface, usage patterns
- **nest-asyncio** (reference file `references/ref-nest-asyncio-reference.md`) — missing: docs URL, API surface, usage patterns
- **networkx** (reference file `references/ref-networkx-reference.md`) — missing: docs URL, API surface, usage patterns
- **nltk** (reference file `references/ref-nltk-reference.md`) — missing: docs URL, API surface, usage patterns
- **nose** (reference file `references/ref-nose-reference.md`) — missing: docs URL, API surface, usage patterns
- **notebook** (reference file `references/ref-notebook-reference.md`) — missing: docs URL, API surface, usage patterns
- **numba** (reference file `references/ref-numba-reference.md`) — missing: docs URL, API surface, usage patterns
- **numdifftools** (reference file `references/ref-numdifftools-reference.md`) — missing: docs URL, API surface, usage patterns
- **numexpr** (reference file `references/ref-numexpr-reference.md`) — missing: docs URL, API surface, usage patterns
- **numpy-base** (reference file `references/ref-numpy-base-reference.md`) — missing: docs URL, API surface, usage patterns
- **numpydoc** (reference file `references/ref-numpydoc-reference.md`) — missing: docs URL, API surface, usage patterns
- **olefile** (reference file `references/ref-olefile-reference.md`) — missing: docs URL, API surface, usage patterns
- **openjpeg** (reference file `references/ref-openjpeg-reference.md`) — missing: docs URL, API surface, usage patterns
- **openpyxl** (reference file `references/ref-openpyxl-reference.md`) — missing: docs URL, API surface, usage patterns
- **openssl** (reference file `references/ref-openssl-reference.md`) — missing: docs URL, API surface, usage patterns
- **osqp** (reference file `references/ref-osqp-reference.md`) — missing: docs URL, API surface, usage patterns
- **packaging** (reference file `references/ref-packaging-reference.md`) — missing: docs URL, API surface, usage patterns
- **pandocfilters** (reference file `references/ref-pandocfilters-reference.md`) — missing: docs URL, API surface, usage patterns
- **panel** (reference file `references/ref-panel-reference.md`) — missing: docs URL, API surface, usage patterns
- **param** (reference file `references/ref-param-reference.md`) — missing: docs URL, API surface, usage patterns
- **parsel** (reference file `references/ref-parsel-reference.md`) — missing: docs URL, API surface, usage patterns
- **parso** (reference file `references/ref-parso-reference.md`) — missing: docs URL, API surface, usage patterns
- **partd** (reference file `references/ref-partd-reference.md`) — missing: docs URL, API surface, usage patterns
- **pathlib** (reference file `references/ref-pathlib-reference.md`) — missing: docs URL, API surface, usage patterns
- **pathspec** (reference file `references/ref-pathspec-reference.md`) — missing: docs URL, API surface, usage patterns
- **patsy** (reference file `references/ref-patsy-reference.md`) — missing: docs URL, API surface, usage patterns
- **pep8** (reference file `references/ref-pep8-reference.md`) — missing: docs URL, API surface, usage patterns
- **pexpect** (reference file `references/ref-pexpect-reference.md`) — missing: docs URL, API surface, usage patterns
- **pickleshare** (reference file `references/ref-pickleshare-reference.md`) — missing: docs URL, API surface, usage patterns
- **pillow** (reference file `references/ref-pillow-reference.md`) — missing: docs URL, API surface, usage patterns
- **pip** (reference file `references/ref-pip-reference.md`) — missing: docs URL, API surface, usage patterns
- **pkce** (reference file `references/ref-pkce-reference.md`) — missing: docs URL, API surface, usage patterns
- **pkginfo** (reference file `references/ref-pkginfo-reference.md`) — missing: docs URL, API surface, usage patterns
- **platformdirs** (reference file `references/ref-platformdirs-reference.md`) — missing: docs URL, API surface, usage patterns
- **pluggy** (reference file `references/ref-pluggy-reference.md`) — missing: docs URL, API surface, usage patterns
- **powerlaw** (reference file `references/ref-powerlaw-reference.md`) — missing: docs URL, API surface, usage patterns
- **powershell_shortcut** (reference file `references/ref-powershellshortcut-reference.md`) — missing: docs URL, API surface, usage patterns
- **poyo** (reference file `references/ref-poyo-reference.md`) — missing: docs URL, API surface, usage patterns
- **proj** (reference file `references/ref-proj-reference.md`) — missing: docs URL, API surface, usage patterns
- **prometheus_client** (reference file `references/ref-prometheusclient-reference.md`) — missing: docs URL, API surface, usage patterns
- **prompt-toolkit** (reference file `references/ref-prompt-toolkit-reference.md`) — missing: docs URL, API surface, usage patterns
- **prompt_toolkit** (reference file `references/ref-prompttoolkit-reference.md`) — missing: docs URL, API surface, usage patterns
- **property-cached** (reference file `references/ref-property-cached-reference.md`) — missing: docs URL, API surface, usage patterns
- **protego** (reference file `references/ref-protego-reference.md`) — missing: docs URL, API surface, usage patterns
- **psutil** (reference file `references/ref-psutil-reference.md`) — missing: docs URL, API surface, usage patterns
- **psycopg2** (reference file `references/ref-psycopg2-reference.md`) — missing: docs URL, API surface, usage patterns
- **ptyprocess** (reference file `references/ref-ptyprocess-reference.md`) — missing: docs URL, API surface, usage patterns
- **pulp** (reference file `references/ref-pulp-reference.md`) — missing: docs URL, API surface, usage patterns
- **py** (reference file `references/ref-py-reference.md`) — missing: docs URL, API surface, usage patterns
- **py-lief** (reference file `references/ref-py-lief-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyarrow** (reference file `references/ref-pyarrow-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyasn1** (reference file `references/ref-pyasn1-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyasn1-modules** (reference file `references/ref-pyasn1-modules-reference.md`) — missing: docs URL, API surface, usage patterns
- **pycodestyle** (reference file `references/ref-pycodestyle-reference.md`) — missing: docs URL, API surface, usage patterns
- **pycosat** (reference file `references/ref-pycosat-reference.md`) — missing: docs URL, API surface, usage patterns
- **pycparser** (reference file `references/ref-pycparser-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyct** (reference file `references/ref-pyct-reference.md`) — missing: docs URL, API surface, usage patterns
- **pycurl** (reference file `references/ref-pycurl-reference.md`) — missing: docs URL, API surface, usage patterns
- **pydantic** (reference file `references/ref-pydantic-reference.md`) — missing: docs URL, API surface, usage patterns
- **pydispatcher** (reference file `references/ref-pydispatcher-reference.md`) — missing: docs URL, API surface, usage patterns
- **pydocstyle** (reference file `references/ref-pydocstyle-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyerfa** (reference file `references/ref-pyerfa-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyflakes** (reference file `references/ref-pyflakes-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyflux** (reference file `references/ref-pyflux-reference.md`) — missing: docs URL, API surface, usage patterns
- **pygments** (reference file `references/ref-pygments-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyhamcrest** (reference file `references/ref-pyhamcrest-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyjwt** (reference file `references/ref-pyjwt-reference.md`) — missing: docs URL, API surface, usage patterns
- **pylint** (reference file `references/ref-pylint-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyls-spyder** (reference file `references/ref-pyls-spyder-reference.md`) — missing: docs URL, API surface, usage patterns
- **pynacl** (reference file `references/ref-pynacl-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyodbc** (reference file `references/ref-pyodbc-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyogrio** (reference file `references/ref-pyogrio-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyopenssl** (reference file `references/ref-pyopenssl-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyparsing** (reference file `references/ref-pyparsing-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyprind** (reference file `references/ref-pyprind-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyproj** (reference file `references/ref-pyproj-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyqt** (reference file `references/ref-pyqt-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyrsistent** (reference file `references/ref-pyrsistent-reference.md`) — missing: docs URL, API surface, usage patterns
- **pysocks** (reference file `references/ref-pysocks-reference.md`) — missing: docs URL, API surface, usage patterns
- **pytables** (reference file `references/ref-pytables-reference.md`) — missing: docs URL, API surface, usage patterns
- **pytest** (reference file `references/ref-pytest-reference.md`) — missing: docs URL, API surface, usage patterns
- **python** (reference file `references/ref-python-reference.md`) — missing: docs URL, API surface, usage patterns
- **python-dateutil** (reference file `references/ref-python-dateutil-reference.md`) — missing: docs URL, API surface, usage patterns
- **python-dotenv** (reference file `references/ref-python-dotenv-reference.md`) — missing: docs URL, API surface, usage patterns
- **python-fastjsonschema** (reference file `references/ref-python-fastjsonschema-reference.md`) — missing: docs URL, API surface, usage patterns
- **python-libarchive-c** (reference file `references/ref-python-libarchive-c-reference.md`) — missing: docs URL, API surface, usage patterns
- **python-lsp-black** (reference file `references/ref-python-lsp-black-reference.md`) — missing: docs URL, API surface, usage patterns
- **python-lsp-jsonrpc** (reference file `references/ref-python-lsp-jsonrpc-reference.md`) — missing: docs URL, API surface, usage patterns
- **python-lsp-server** (reference file `references/ref-python-lsp-server-reference.md`) — missing: docs URL, API surface, usage patterns
- **python-slugify** (reference file `references/ref-python-slugify-reference.md`) — missing: docs URL, API surface, usage patterns
- **python-snappy** (reference file `references/ref-python-snappy-reference.md`) — missing: docs URL, API surface, usage patterns
- **python_abi** (reference file `references/ref-pythonabi-reference.md`) — missing: docs URL, API surface, usage patterns
- **pytz** (reference file `references/ref-pytz-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyusda** (reference file `references/ref-pyusda-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyviz_comms** (reference file `references/ref-pyvizcomms-reference.md`) — missing: docs URL, API surface, usage patterns
- **pywavelets** (reference file `references/ref-pywavelets-reference.md`) — missing: docs URL, API surface, usage patterns
- **pywin32** (reference file `references/ref-pywin32-reference.md`) — missing: docs URL, API surface, usage patterns
- **pywin32-ctypes** (reference file `references/ref-pywin32-ctypes-reference.md`) — missing: docs URL, API surface, usage patterns
- **pywinpty** (reference file `references/ref-pywinpty-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyyaml** (reference file `references/ref-pyyaml-reference.md`) — missing: docs URL, API surface, usage patterns
- **pyzmq** (reference file `references/ref-pyzmq-reference.md`) — missing: docs URL, API surface, usage patterns
- **qdarkstyle** (reference file `references/ref-qdarkstyle-reference.md`) — missing: docs URL, API surface, usage patterns
- **qdldl** (reference file `references/ref-qdldl-reference.md`) — missing: docs URL, API surface, usage patterns
- **qstylizer** (reference file `references/ref-qstylizer-reference.md`) — missing: docs URL, API surface, usage patterns
- **qt** (reference file `references/ref-qt-reference.md`) — missing: docs URL, API surface, usage patterns
- **qtawesome** (reference file `references/ref-qtawesome-reference.md`) — missing: docs URL, API surface, usage patterns
- **qtconsole** (reference file `references/ref-qtconsole-reference.md`) — missing: docs URL, API surface, usage patterns
- **qtpy** (reference file `references/ref-qtpy-reference.md`) — missing: docs URL, API surface, usage patterns
- **queuelib** (reference file `references/ref-queuelib-reference.md`) — missing: docs URL, API surface, usage patterns
- **random-dict** (reference file `references/ref-random-dict-reference.md`) — missing: docs URL, API surface, usage patterns
- **randomdict** (reference file `references/ref-randomdict-reference.md`) — missing: docs URL, API surface, usage patterns
- **re2** (reference file `references/ref-re2-reference.md`) — missing: docs URL, API surface, usage patterns
- **regex** (reference file `references/ref-regex-reference.md`) — missing: docs URL, API surface, usage patterns
- **requests** (reference file `references/ref-requests-reference.md`) — missing: docs URL, API surface, usage patterns
- **requests-file** (reference file `references/ref-requests-file-reference.md`) — missing: docs URL, API surface, usage patterns
- **retrying** (reference file `references/ref-retrying-reference.md`) — missing: docs URL, API surface, usage patterns
- **rope** (reference file `references/ref-rope-reference.md`) — missing: docs URL, API surface, usage patterns
- **rtree** (reference file `references/ref-rtree-reference.md`) — missing: docs URL, API surface, usage patterns
- **ruamel.yaml** (reference file `references/ref-ruamelyaml-reference.md`) — missing: docs URL, API surface, usage patterns
- **ruamel.yaml.clib** (reference file `references/ref-ruamelyamlclib-reference.md`) — missing: docs URL, API surface, usage patterns
- **ruamel_yaml** (reference file `references/ref-ruamelyaml-reference.md`) — missing: docs URL, API surface, usage patterns
- **s3transfer** (reference file `references/ref-s3transfer-reference.md`) — missing: docs URL, API surface, usage patterns
- **scikit-image** (reference file `references/ref-scikit-image-reference.md`) — missing: docs URL, API surface, usage patterns
- **scikit-learn** (reference file `references/ref-scikit-learn-reference.md`) — missing: docs URL, API surface, usage patterns
- **scikit-learn-intelex** (reference file `references/ref-scikit-learn-intelex-reference.md`) — missing: docs URL, API surface, usage patterns
- **scipy** (reference file `references/ref-scipy-reference.md`) — missing: docs URL, API surface, usage patterns
- **scrapy** (reference file `references/ref-scrapy-reference.md`) — missing: docs URL, API surface, usage patterns
- **scs** (reference file `references/ref-scs-reference.md`) — missing: docs URL, API surface, usage patterns
- **seaborn** (reference file `references/ref-seaborn-reference.md`) — missing: docs URL, API surface, usage patterns
- **semver** (reference file `references/ref-semver-reference.md`) — missing: docs URL, API surface, usage patterns
- **send2trash** (reference file `references/ref-send2trash-reference.md`) — missing: docs URL, API surface, usage patterns
- **service_identity** (reference file `references/ref-serviceidentity-reference.md`) — missing: docs URL, API surface, usage patterns
- **setuptools** (reference file `references/ref-setuptools-reference.md`) — missing: docs URL, API surface, usage patterns
- **shapely** (reference file `references/ref-shapely-reference.md`) — missing: docs URL, API surface, usage patterns
- **sip** (reference file `references/ref-sip-reference.md`) — missing: docs URL, API surface, usage patterns
- **six** (reference file `references/ref-six-reference.md`) — missing: docs URL, API surface, usage patterns
- **smart_open** (reference file `references/ref-smartopen-reference.md`) — missing: docs URL, API surface, usage patterns
- **snappy** (reference file `references/ref-snappy-reference.md`) — missing: docs URL, API surface, usage patterns
- **sniffio** (reference file `references/ref-sniffio-reference.md`) — missing: docs URL, API surface, usage patterns
- **snowballstemmer** (reference file `references/ref-snowballstemmer-reference.md`) — missing: docs URL, API surface, usage patterns
- **sortedcollections** (reference file `references/ref-sortedcollections-reference.md`) — missing: docs URL, API surface, usage patterns
- **sortedcontainers** (reference file `references/ref-sortedcontainers-reference.md`) — missing: docs URL, API surface, usage patterns
- **soupsieve** (reference file `references/ref-soupsieve-reference.md`) — missing: docs URL, API surface, usage patterns
- **sphinx** (reference file `references/ref-sphinx-reference.md`) — missing: docs URL, API surface, usage patterns
- **sphinxcontrib-applehelp** (reference file `references/ref-sphinxcontrib-applehelp-reference.md`) — missing: docs URL, API surface, usage patterns
- **sphinxcontrib-devhelp** (reference file `references/ref-sphinxcontrib-devhelp-reference.md`) — missing: docs URL, API surface, usage patterns
- **sphinxcontrib-htmlhelp** (reference file `references/ref-sphinxcontrib-htmlhelp-reference.md`) — missing: docs URL, API surface, usage patterns
- **sphinxcontrib-jsmath** (reference file `references/ref-sphinxcontrib-jsmath-reference.md`) — missing: docs URL, API surface, usage patterns
- **sphinxcontrib-qthelp** (reference file `references/ref-sphinxcontrib-qthelp-reference.md`) — missing: docs URL, API surface, usage patterns
- **sphinxcontrib-serializinghtml** (reference file `references/ref-sphinxcontrib-serializinghtml-reference.md`) — missing: docs URL, API surface, usage patterns
- **spyder** (reference file `references/ref-spyder-reference.md`) — missing: docs URL, API surface, usage patterns
- **spyder-kernels** (reference file `references/ref-spyder-kernels-reference.md`) — missing: docs URL, API surface, usage patterns
- **sqlalchemy** (reference file `references/ref-sqlalchemy-reference.md`) — missing: docs URL, API surface, usage patterns
- **statsmodels** (reference file `references/ref-statsmodels-reference.md`) — missing: docs URL, API surface, usage patterns
- **sympy** (reference file `references/ref-sympy-reference.md`) — missing: docs URL, API surface, usage patterns
- **tabulate** (reference file `references/ref-tabulate-reference.md`) — missing: docs URL, API surface, usage patterns
- **tbb** (reference file `references/ref-tbb-reference.md`) — missing: docs URL, API surface, usage patterns
- **tbb4py** (reference file `references/ref-tbb4py-reference.md`) — missing: docs URL, API surface, usage patterns
- **tblib** (reference file `references/ref-tblib-reference.md`) — missing: docs URL, API surface, usage patterns
- **tenacity** (reference file `references/ref-tenacity-reference.md`) — missing: docs URL, API surface, usage patterns
- **terminado** (reference file `references/ref-terminado-reference.md`) — missing: docs URL, API surface, usage patterns
- **testpath** (reference file `references/ref-testpath-reference.md`) — missing: docs URL, API surface, usage patterns
- **text-unidecode** (reference file `references/ref-text-unidecode-reference.md`) — missing: docs URL, API surface, usage patterns
- **textdistance** (reference file `references/ref-textdistance-reference.md`) — missing: docs URL, API surface, usage patterns
- **threadpoolctl** (reference file `references/ref-threadpoolctl-reference.md`) — missing: docs URL, API surface, usage patterns
- **three-merge** (reference file `references/ref-three-merge-reference.md`) — missing: docs URL, API surface, usage patterns
- **tifffile** (reference file `references/ref-tifffile-reference.md`) — missing: docs URL, API surface, usage patterns
- **tiledb** (reference file `references/ref-tiledb-reference.md`) — missing: docs URL, API surface, usage patterns
- **tinycss** (reference file `references/ref-tinycss-reference.md`) — missing: docs URL, API surface, usage patterns
- **tk** (reference file `references/ref-tk-reference.md`) — missing: docs URL, API surface, usage patterns
- **tldextract** (reference file `references/ref-tldextract-reference.md`) — missing: docs URL, API surface, usage patterns
- **toml** (reference file `references/ref-toml-reference.md`) — missing: docs URL, API surface, usage patterns
- **tomli** (reference file `references/ref-tomli-reference.md`) — missing: docs URL, API surface, usage patterns
- **tomlkit** (reference file `references/ref-tomlkit-reference.md`) — missing: docs URL, API surface, usage patterns
- **toolz** (reference file `references/ref-toolz-reference.md`) — missing: docs URL, API surface, usage patterns
- **tornado** (reference file `references/ref-tornado-reference.md`) — missing: docs URL, API surface, usage patterns
- **tqdm** (reference file `references/ref-tqdm-reference.md`) — missing: docs URL, API surface, usage patterns
- **traitlets** (reference file `references/ref-traitlets-reference.md`) — missing: docs URL, API surface, usage patterns
- **twisted** (reference file `references/ref-twisted-reference.md`) — missing: docs URL, API surface, usage patterns
- **twisted-iocpsupport** (reference file `references/ref-twisted-iocpsupport-reference.md`) — missing: docs URL, API surface, usage patterns
- **typing-extensions** (reference file `references/ref-typing-extensions-reference.md`) — missing: docs URL, API surface, usage patterns
- **typing_extensions** (reference file `references/ref-typingextensions-reference.md`) — missing: docs URL, API surface, usage patterns
- **tzdata** (reference file `references/ref-tzdata-reference.md`) — missing: docs URL, API surface, usage patterns
- **ujson** (reference file `references/ref-ujson-reference.md`) — missing: docs URL, API surface, usage patterns
- **unidecode** (reference file `references/ref-unidecode-reference.md`) — missing: docs URL, API surface, usage patterns
- **urllib3** (reference file `references/ref-urllib3-reference.md`) — missing: docs URL, API surface, usage patterns
- **utf8proc** (reference file `references/ref-utf8proc-reference.md`) — missing: docs URL, API surface, usage patterns
- **vc** (reference file `references/ref-vc-reference.md`) — missing: docs URL, API surface, usage patterns
- **vs2015_runtime** (reference file `references/ref-vs2015runtime-reference.md`) — missing: docs URL, API surface, usage patterns
- **w3lib** (reference file `references/ref-w3lib-reference.md`) — missing: docs URL, API surface, usage patterns
- **watchdog** (reference file `references/ref-watchdog-reference.md`) — missing: docs URL, API surface, usage patterns
- **wcwidth** (reference file `references/ref-wcwidth-reference.md`) — missing: docs URL, API surface, usage patterns
- **webencodings** (reference file `references/ref-webencodings-reference.md`) — missing: docs URL, API surface, usage patterns
- **websocket-client** (reference file `references/ref-websocket-client-reference.md`) — missing: docs URL, API surface, usage patterns
- **werkzeug** (reference file `references/ref-werkzeug-reference.md`) — missing: docs URL, API surface, usage patterns
- **wheel** (reference file `references/ref-wheel-reference.md`) — missing: docs URL, API surface, usage patterns
- **widgetsnbextension** (reference file `references/ref-widgetsnbextension-reference.md`) — missing: docs URL, API surface, usage patterns
- **win_inet_pton** (reference file `references/ref-wininetpton-reference.md`) — missing: docs URL, API surface, usage patterns
- **win_unicode_console** (reference file `references/ref-winunicodeconsole-reference.md`) — missing: docs URL, API surface, usage patterns
- **wincertstore** (reference file `references/ref-wincertstore-reference.md`) — missing: docs URL, API surface, usage patterns
- **winpty** (reference file `references/ref-winpty-reference.md`) — missing: docs URL, API surface, usage patterns
- **wrapt** (reference file `references/ref-wrapt-reference.md`) — missing: docs URL, API surface, usage patterns
- **xarray** (reference file `references/ref-xarray-reference.md`) — missing: docs URL, API surface, usage patterns
- **xerces-c** (reference file `references/ref-xerces-c-reference.md`) — missing: docs URL, API surface, usage patterns
- **xlrd** (reference file `references/ref-xlrd-reference.md`) — missing: docs URL, API surface, usage patterns
- **xlsxwriter** (reference file `references/ref-xlsxwriter-reference.md`) — missing: docs URL, API surface, usage patterns
- **xlwings** (reference file `references/ref-xlwings-reference.md`) — missing: docs URL, API surface, usage patterns
- **xyzservices** (reference file `references/ref-xyzservices-reference.md`) — missing: docs URL, API surface, usage patterns
- **xz** (reference file `references/ref-xz-reference.md`) — missing: docs URL, API surface, usage patterns
- **yaml** (reference file `references/ref-yaml-reference.md`) — missing: docs URL, API surface, usage patterns
- **yapf** (reference file `references/ref-yapf-reference.md`) — missing: docs URL, API surface, usage patterns
- **yfinance** (reference file `references/ref-yfinance-reference.md`) — missing: docs URL, API surface, usage patterns
- **zeromq** (reference file `references/ref-zeromq-reference.md`) — missing: docs URL, API surface, usage patterns
- **zfp** (reference file `references/ref-zfp-reference.md`) — missing: docs URL, API surface, usage patterns
- **zict** (reference file `references/ref-zict-reference.md`) — missing: docs URL, API surface, usage patterns
- **zipp** (reference file `references/ref-zipp-reference.md`) — missing: docs URL, API surface, usage patterns
- **zlib** (reference file `references/ref-zlib-reference.md`) — missing: docs URL, API surface, usage patterns
- **zope** (reference file `references/ref-zope-reference.md`) — missing: docs URL, API surface, usage patterns
- **zope.interface** (reference file `references/ref-zopeinterface-reference.md`) — missing: docs URL, API surface, usage patterns
- **zstandard** (reference file `references/ref-zstandard-reference.md`) — missing: docs URL, API surface, usage patterns
- **zstd** (reference file `references/ref-zstd-reference.md`) — missing: docs URL, API surface, usage patterns

If this list reads "No tools with missing metadata", your work is complete — return to `@orchestrator`.

---

<!-- AGENTTEAMS:BEGIN memory_index_consultation v=1 -->
## Memory-index consultation *(applies when `references/memory-index.json` is present)*

Before opening external documentation tiers, check whether the team has already researched this tool — prior handoffs, work summaries, or tool reference files may already carry the `docs_url`, `api_surface`, or version-pinned `common_patterns` for the version listed in the project brief:

```bash
agentteams --query-index "<tool name> <version>" --query-strategy lexical --query-k 5 --description .agentteams/brief.json --project . --output .github/agents --no-scan --yes
```

If a prior research artifact surfaces (top-1 ≥ 3.0 is a reliable hit; 1.0–3.0 is candidate-for-inspection, responsive snippet), open it and reuse the verified fields — re-verifying only the `docs_url` against the live site to confirm it has not moved. Cite the prior artifact in your output so `@agent-updater` knows the data was reused, not re-fetched. Never block on the index; if absent/empty, proceed to Tier 1 below.
<!-- AGENTTEAMS:END memory_index_consultation -->

## Documentation Discovery Strategies

Work through these strategies in order for each tool. Stop at the first tier that yields a verifiable official source.

### Tier 1 — Official Sources (Always Try First)

1. **Official Documentation Site**
   - Search `<tool-name> official documentation` or visit `docs.<tool-name>.org`, `<tool-name>.dev/docs`, or `<tool-name>.io/docs`.
   - Confirm the page describes the tool's own public API, not a third-party tutorial or commentary.

2. **Package Registry Pages**
   - Python: `https://pypi.org/project/<package-name>/` → check "Project links" section for the documentation URL.
   - JavaScript / TypeScript: `https://www.npmjs.com/package/<package-name>` → check "Homepage" link.
   - Rust: `https://docs.rs/<crate-name>/latest/` — auto-generated from source; authoritative for all Rust crates.
   - R: `https://cran.r-project.org/package=<pkg-name>` → check "Reference manual" PDF.
   - Julia: `https://juliahub.com/ui/Packages/<PackageName>` → follow the documentation link.

3. **GitHub Releases and README**
   - Navigate to the canonical upstream GitHub repository.
   - Locate the documentation URL in the README "Documentation" badge or link.
   - Check `https://github.com/<org>/<repo>/releases/latest` for the current version and changelog.

### Tier 2 — Structured Reference Sources (Use When Tier 1 is Incomplete)

4. **ReadTheDocs**
   - URL pattern: `https://<package-name>.readthedocs.io/en/stable/`
   - Common for Python scientific stack, data engineering tools, and ML frameworks.

5. **GitHub Pages Doc Sites**
   - URL pattern: `https://<org>.github.io/<repo>/`
   - Typical for JavaScript / TypeScript libraries using TypeDoc or Docusaurus.

6. **MDN Web Docs** (browser-native and Web APIs only)
   - URL: `https://developer.mozilla.org/en-US/docs/Web/API/<InterfaceName>`
   - Authoritative for Web APIs (Fetch, WebSocket, Web Audio API, Web MIDI API, etc.).

7. **W3C and WHATWG Specifications**
   - Use for browser web standards when MDN is incomplete on edge cases.
   - W3C: `https://www.w3.org/TR/<spec-name>/`
   - WHATWG: `https://html.spec.whatwg.org/`

### Tier 3 — Verification Fallbacks (Use Only When No Official Source Exists)

8. **Verified Repository README**
   - Only valid if the README is in the canonical upstream repository and explicitly states version compatibility.
   - Do not treat "Examples" or "Quickstart" sections as a substitute for a full API surface.

9. **Release Notes / Changelog**
   - Use to confirm the current version and identify deprecated APIs.
   - Changelogs describe deltas only — never use as the primary API surface reference.

---

## What to Research Per Tool

For each tool in the list above, determine:

| Field | What to Produce | Acceptable Source Tier |
|-------|----------------|------------------------|
| `docs_url` | Canonical documentation URL — versioned if available (e.g., `.../en/v3.2/`) | Tier 1 only |
| `api_surface` | 3–8 key classes, functions, or CLI commands the project code directly depends on | Tier 1 or 2 |
| `common_patterns` | 2–4 usage patterns and pitfalls specific to the tool version and use case | Tier 1 or 2; Tier 3 only with explicit citation |

---

## Quality Constraints

> ⛔ **These constraints are non-negotiable.**

1. **Never fabricate a URL.** Every `docs_url` must be content you have read. If a URL returns 404 or redirects to an unrelated page, discard it and try the next strategy.

2. **Do not use tutorial sites as primary sources.** `medium.com`, `dev.to`, `stackoverflow.com`, `digitalocean.com`, `geeksforgeeks.org`, and similar tutorial or Q&A sites are not authoritative.

3. **Version accuracy is mandatory.** Record `api_surface` and `common_patterns` for the version listed in the project brief, not the latest version if they differ.

4. **Cite your source tier.** Add an inline parenthetical `(Tier 2: <url>)` after any `api_surface` or `common_patterns` entry derived from Tier 2 or 3 sources.

5. **Scope discipline.** Research only the tools in the list above. Do not expand scope to other project dependencies.

---

## Output Format

For each tool, produce a fenced block:

```
Tool: <tool-name> <version>
docs_url: <verified URL>
api_surface: |
  <key class, function, or command 1>
  <key class, function, or command 2>
  ...
common_patterns: |
  <usage pattern or pitfall 1>
  <usage pattern or pitfall 2>
  ...
```

After completing all tools in the list, hand off to `@agent-updater` with these instructions:

1. Add `docs_url`, `api_surface`, and `common_patterns` to each matching tool entry in the project brief so that future pipeline reruns auto-populate these fields.
2. Directly update the affected tool documents — reference files in `references/` and Claude skills in `.claude/skills/` — so the current generation is complete without requiring a full rerender.

## Project-Specific Notes

> ⚙️ **USER-EDITABLE** — project-specific rules, overrides, and extensions for this agent. This section lies outside every `AGENTTEAMS` fence and is preserved verbatim across `agentteams --update --merge`.
