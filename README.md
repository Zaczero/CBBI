# ColinTalksCrypto Bitcoin Bull Run Index (CBBI)

An official Python implementation of the **ColinTalksCrypto Bitcoin Bull Run Index** (CBBI).

The CBBI is a Bitcoin index that utilizes advanced, real-time analysis of 12 metrics to help us understand what stage of the Bitcoin bull run and bear market cycles we are in.
The CBBI is time-independent and price-independent.
It simply indicates whether it believes we are approaching the top/bottom of a Bitcoin cycle.

If you want to learn more, [watch this video](https://www.youtube.com/watch?v=bq7djf1n0j4).

## Visit our website

Bookmark it to receive latest CBBI updates.

- [CBBI.info](https://cbbi.info/)

## Check out this demo

[![asciicast](https://asciinema.org/a/6oYLls2F1nCz6Sv6KyMKLL0n7.svg)](https://asciinema.org/a/6oYLls2F1nCz6Sv6KyMKLL0n7)

## Usage

*Recommended Python version: 3.9*

### Install required packages

`$ pip install -r requirements.txt`

### Execute the script

`$ py main.py --help`  
`$ py main.py`

## Metrics

The current CBBI version *(October 2021)* includes the following metrics:

| Name | Link |
|---------------|------|
| The Golden 51%-49% Ratio | [Visit page](https://www.tradingview.com/chart/BTCUSD/QBeNL8jt-BITCOIN-The-Golden-51-49-Ratio-600-days-of-Bull-Market-left/) |
| Google Trends for "Bitcoin" | [Visit page](https://trends.google.com/trends/explore?date=today%205-y&q=bitcoin) |
| Halving-to-Peak Days | [Visit page](https://www.youtube.com/watch?v=oxR_0njPht8&t=290s) |
| Pi Cycle Top Indicator | [Visit page](https://www.lookintobitcoin.com/charts/pi-cycle-top-indicator/) |
| 2 Year Moving Average | [Visit page](https://www.lookintobitcoin.com/charts/bitcoin-investor-tool/) |
| Bitcoin Trolololo Trend Line | [Visit page](https://www.blockchaincenter.net/bitcoin-rainbow-chart/) |
| RUPL/NUPL Chart | [Visit page](https://www.lookintobitcoin.com/charts/relative-unrealized-profit--loss/) |
| Puell Multiple | [Visit page](https://www.lookintobitcoin.com/charts/puell-multiple/) |
| MVRV Z-Score | [Visit page](https://www.lookintobitcoin.com/charts/mvrv-zscore/) |
| RHODL Ratio | [Visit page](https://www.lookintobitcoin.com/charts/rhodl-ratio/) |
| Reserve Risk | [Visit page](https://www.lookintobitcoin.com/charts/reserve-risk/) |
| Woobull Top Cap vs CVDD | [Visit page](https://charts.woobull.com/bitcoin-price-models/) |

## Environment variables

This project supports `.env` files which provide a convenient way of setting environment variables.

### GOOGLE_PROXY

Defines a requests-supported proxy string used during Google Trends metric calculation.
If unset or empty, a direct connection will be made.
May be used to resolve the `Google returned a response with code 429` issue.

#### Example usage

* GOOGLE_PROXY=https://host:port
* GOOGLE_PROXY=https://user:pass@host:port
