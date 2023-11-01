# ColinTalksCrypto Bitcoin Bull Run Index (CBBI)

![GitHub Pipenv locked Python version](https://shields.monicz.dev/github/pipenv/locked/python-version/Zaczero/CBBI)
[![Project license](https://shields.monicz.dev/github/license/Zaczero/CBBI)](https://github.com/Zaczero/CBBI/blob/main/LICENSE)
[![GitHub Repo stars](https://shields.monicz.dev/github/stars/Zaczero/CBBI?style=social)](https://github.com/Zaczero/CBBI)

The official Python implementation of the **ColinTalksCrypto Bitcoin Bull Run Index** (CBBI).

The CBBI is a Bitcoin index that utilizes advanced, real-time analysis of 10 metrics
to help us understand what stage of the Bitcoin bull run and bear market cycles we are in.
The CBBI is time-independent and price-independent.
It simply indicates whether it believes we are approaching the top/bottom of a Bitcoin cycle.

If you want to learn more, [check out this video](https://www.youtube.com/watch?v=bq7djf1n0j4).

## Visit our website

Bookmark it and receive latest CBBI updates.

- [CBBI.info](https://cbbi.info/)

## Script Demo

[![asciicast](https://raw.githubusercontent.com/Zaczero/CBBI/main/asciinema/thumbnail.webp)](https://asciinema.org/a/KFkbKPULf9PGvY8Fmh4QLn0FE)

## Docker Usage

To use the CBBI script with Docker, run the following command:

```sh
$ docker run --rm --pull=always zaczero/cbbi --help
$ docker run --rm --pull=always zaczero/cbbi
```

## Manual Usage

To use the CBBI script without Docker, follow these steps:

_Recommended Python version: 3.11_

### 1. Install pipenv:

[Pipenv: Python Dev Workflow for Humans](https://pipenv.pypa.io/en/latest/#install-pipenv-today)

```sh
# TLDR;
$ pip install --user pipenv
```

### 2. Install required packages:

**NOTE:** The `pipenv` commands should be executed within the project directory.

```sh
$ pipenv install
```

### 3. Run the script:

```sh
$ pipenv run python main.py --help
$ pipenv run python main.py
```

#### or _(using pipenv shell)_

```sh
$ pipenv shell
$> python main.py --help
$> python main.py
```

## Metrics

The current CBBI version _(November 2022)_ includes the following metrics:

| Name                         | Link                                                                  |
| ---------------------------- | --------------------------------------------------------------------- |
| Pi Cycle Top Indicator       | [Visit page](https://coinsoto.com/indexdata/piCycleTop)               |
| RUPL/NUPL Chart              | [Visit page](https://coinsoto.com/indexdata/realizedProf)             |
| RHODL Ratio                  | [Visit page](https://coinsoto.com/indexdata/rhodlRatio)               |
| Puell Multiple               | [Visit page](https://coinsoto.com/indexdata/puellMultiple)            |
| 2 Year Moving Average        | [Visit page](https://coinsoto.com/indexdata/year2MA)                  |
| Bitcoin Trolololo Trend Line | [Visit page](https://www.blockchaincenter.net/bitcoin-rainbow-chart/) |
| MVRV Z-Score                 | [Visit page](https://coinsoto.com/indexdata/score)                    |
| Reserve Risk                 | [Visit page](https://coinsoto.com/indexdata/reserveRisk)              |
| Woobull Top Cap vs CVDD      | [Visit page](https://charts.woobull.com/bitcoin-price-models/)        |

## Environment Variables

This project supports `.env` files, which provide a convenient way of setting environment variables.

To use this feature, create a file called `.env` in the project's root directory,
and add environment variables in the following format:

```sh
VARIABLE_NAME=value
```

### GLASSNODE_API_KEY

Defines an API key to be used during GlassNode fallback requests.
If unset or empty, a cache fallback will be used instead (via CBBI.info).

#### Example usage

- GLASSNODE_API_KEY=REPLACE_ME

### TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

Define both variables to receive Telegram notifications about metric errors that occur during the execution.

#### Example usage

- TELEGRAM_TOKEN=REPLACE_ME
- TELEGRAM_CHAT_ID=123456

## Footer

### Contact me

https://monicz.dev/#get-in-touch

### License

This project is licensed under the GNU Affero General Public License v3.0.

The complete license text can be accessed in the repository at [LICENSE](https://github.com/Zaczero/CBBI/blob/main/LICENSE).
