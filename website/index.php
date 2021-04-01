<?php
define("DATA_FILE", "latest");
define("DATE_FORMAT", "F jS, Y");

if (!file_exists(DATA_FILE)) {
    http_response_code(500);
    exit("500: data file not found");
}

require_once __DIR__."/vendor/autoload.php";

$highlighter = new \AnsiEscapesToHtml\Highlighter();

$output_raw = file_get_contents(DATA_FILE);
$output_raw = str_replace(" ", "&nbsp;", $output_raw);
$output = $highlighter->toHtml($output_raw);

$mtime = filemtime(DATA_FILE);
$updated_on = date(DATE_FORMAT, $mtime);
?><!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, shrink-to-fit=yes">
        
        <link rel="shortcut icon" href="/favicon.ico">
        <link rel="icon" sizes="16x16 32x32 64x64" href="/favicon.ico">
        <link rel="icon" type="image/png" sizes="196x196" href="/favicon-192.png">
        <link rel="icon" type="image/png" sizes="160x160" href="/favicon-160.png">
        <link rel="icon" type="image/png" sizes="96x96" href="/favicon-96.png">
        <link rel="icon" type="image/png" sizes="64x64" href="/favicon-64.png">
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16.png">
        <link rel="apple-touch-icon" href="/favicon-57.png">
        <link rel="apple-touch-icon" sizes="114x114" href="/favicon-114.png">
        <link rel="apple-touch-icon" sizes="72x72" href="/favicon-72.png">
        <link rel="apple-touch-icon" sizes="144x144" href="/favicon-144.png">
        <link rel="apple-touch-icon" sizes="60x60" href="/favicon-60.png">
        <link rel="apple-touch-icon" sizes="120x120" href="/favicon-120.png">
        <link rel="apple-touch-icon" sizes="76x76" href="/favicon-76.png">
        <link rel="apple-touch-icon" sizes="152x152" href="/favicon-152.png">
        <link rel="apple-touch-icon" sizes="180x180" href="/favicon-180.png">
        <meta name="msapplication-TileColor" content="#FFFFFF">
        <meta name="msapplication-TileImage" content="/favicon-144.png">
        <meta name="msapplication-config" content="/browserconfig.xml">

        <title>CBBI - Colin Talks Crypto Bitcoin Bull Run Index</title>
        <meta name="description" content="Python implementation of the Colin Talks Crypto Bitcoin Bull Run Index. It makes use of multiple metrics to evaluate the confidence that we are at the peak of the current Bitcoin's bull run." />

        <style>
        html, body {
            padding: 16px;
            background-color: black;
            font-size: 18px;
            letter-spacing: .25px;
            line-height: 1.4;
        }
        </style>
    </head>
    <body>
        <code>
            <?=$output?>
            <span style="color: lightgray; font-style: italic;">Updated on <?=$updated_on?></span>
        </code>
    </body>
</html>
