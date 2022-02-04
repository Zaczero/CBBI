# ColinTalksCrypto Bitcoin Bull Run Index (CBBI)

![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/Zaczero/CBBI)
[![GitHub](https://img.shields.io/github/license/Zaczero/CBBI)](https://github.com/Zaczero/CBBI/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/Zaczero/CBBI?style=social)](https://github.com/Zaczero/CBBI)

The official Python implementation of the **ColinTalksCrypto Bitcoin Bull Run Index** (CBBI).

The CBBI is a Bitcoin index that utilizes advanced, real-time analysis of 11 metrics to help us understand what stage of the Bitcoin bull run and bear market cycles we are in.
The CBBI is time-independent and price-independent.
It simply indicates whether it believes we are approaching the top/bottom of a Bitcoin cycle.

If you want to learn more, [check out this video](https://www.youtube.com/watch?v=bq7djf1n0j4).

## Visit our website

Bookmark it to receive latest CBBI updates.

- [CBBI.info](https://cbbi.info/)

## Script demo

[![asciicast](https://raw.githubusercontent.com/Zaczero/CBBI/main/asciinema/thumbnail.webp)](https://asciinema.org/a/KFkbKPULf9PGvY8Fmh4QLn0FE)

## Docker usage

### Pull the image

`$ docker pull zaczero/cbbi`

### Execute the script

`$ docker run zaczero/cbbi --help`  
`$ docker run zaczero/cbbi`

## Manual usage

*Recommended Python version: 3.9*

### Install pipenv

[Pipenv: Python Dev Workflow for Humans](https://pipenv.pypa.io/en/latest/#install-pipenv-today)

### Install required packages

**NOTE:** The `pipenv` commands shall be executed within the project directory.

`$ pipenv install`

### Execute the script

`$ pipenv run python main.py --help`  
`$ pipenv run python main.py`

#### or

`$ pipenv shell`  
`$> python main.py --help`  
`$> python main.py`

## Metrics table

The current CBBI version *(November 2021)* includes the following metrics:

| Name                         | Link                                                                                   |
|------------------------------|----------------------------------------------------------------------------------------|
| Pi Cycle Top Indicator       | [Visit page](https://www.lookintobitcoin.com/charts/pi-cycle-top-indicator/)           |
| RUPL/NUPL Chart              | [Visit page](https://www.lookintobitcoin.com/charts/relative-unrealized-profit--loss/) |
| RHODL Ratio                  | [Visit page](https://www.lookintobitcoin.com/charts/rhodl-ratio/)                      |
| Puell Multiple               | [Visit page](https://www.lookintobitcoin.com/charts/puell-multiple/)                   |
| 2 Year Moving Average        | [Visit page](https://www.lookintobitcoin.com/charts/bitcoin-investor-tool/)            |
| Bitcoin Trolololo Trend Line | [Visit page](https://www.blockchaincenter.net/bitcoin-rainbow-chart/)                  |
| MVRV Z-Score                 | [Visit page](https://www.lookintobitcoin.com/charts/mvrv-zscore/)                      |
| Reserve Risk                 | [Visit page](https://www.lookintobitcoin.com/charts/reserve-risk/)                     |
| Woobull Top Cap vs CVDD      | [Visit page](https://charts.woobull.com/bitcoin-price-models/)                         |
| Halving-to-Peak Days         | [Visit page](https://www.youtube.com/watch?v=oxR_0njPht8&t=290s)                       |
| Google Trends for "Bitcoin"  | [Visit page](https://trends.google.com/trends/explore?date=today%205-y&q=bitcoin)      |

## Environment variables

This project supports `.env` files which provide a convenient way of setting environment variables.

### GOOGLE_PROXY

Defines a requests-supported proxy string used during Google Trends metric calculation.
If unset or empty, a direct connection will be made.
It may be used to resolve the `Google returned a response with code 429` issue.

#### Example usage

* GOOGLE_PROXY=https://host:port
* GOOGLE_PROXY=https://user:pass@host:port

### GLASSNODE_API_KEY

Defines an API key to be used during GlassNode fallback requests.
If unset or empty, a cache fallback will be used instead (via CBBI.info).

#### Example usage

* GLASSNODE_API_KEY=REPLACE_ME

## Footer

### Contact

* Email: [kamil@monicz.pl](mailto:kamil@monicz.pl)
* LinkedIn: [linkedin.com/in/kamil-monicz](https://www.linkedin.com/in/kamil-monicz/)
