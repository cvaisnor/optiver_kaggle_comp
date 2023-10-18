# !/bin/sh

set -ex
pushd kaggle/input/optiver-trading-at-the-close
kaggle competitions download -c optiver-trading-at-the-close
unzip optiver-trading-at-the-close.zip
rm optiver-trading-at-the-close.zip
popd
