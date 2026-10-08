# Model 1 EDA summary (training split)

## Class balance (train, %)

```
result
H    46.0
D    25.2
A    28.8
```

## Goal counts vs Poisson (%)

```
        home                 away          
  observed % Poisson % observed % Poisson %
0       22.7      21.4       33.5      31.3
1       32.3      33.0       34.5      36.4
2       24.5      25.4       19.6      21.1
3       12.7      13.1        8.4       8.2
4        5.4       5.0        2.8       2.4
5        1.6       1.6        0.9       0.5
6        0.6       0.4        0.2       0.1
7        0.2       0.1        0.0       0.0
```

## Low scores and draws vs Poisson (%)

```
           observed %  Poisson %
score                           
0-0              7.53       8.00
1-0             10.25      10.61
0-1              7.14       8.43
1-1             11.65      10.76
all draws       25.31      24.00
```

## Home advantage by league

```
result                        H      D      A  home goals  away goals
league                                                               
Scotland Premier League   40.61  24.57  34.82        1.40        1.22
Portugal Liga ZON Sagres  44.04  26.21  29.75        1.39        1.11
France Ligue 1            45.04  28.70  26.26        1.42        1.03
Switzerland Super League  45.18  24.77  30.05        1.62        1.25
Poland Ekstraklasa        45.57  27.62  26.82        1.40        1.01
Germany 1. Bundesliga     45.81  24.42  29.77        1.63        1.27
Italy Serie A             46.22  26.02  27.77        1.49        1.14
England Premier League    46.24  24.69  29.06        1.57        1.17
Belgium Jupiler League    47.83  24.37  27.80        1.61        1.18
Netherlands Eredivisie    47.96  23.98  28.06        1.80        1.31
Spain LIGA BBVA           49.20  22.06  28.74        1.61        1.13
```

## Result rate (%) by pre-match rating difference

```
result         H     D     A  matches
bin                                  
< -1        16.7  20.4  63.0     1158
-1..-0.6    28.1  26.2  45.7     1632
-0.6..-0.3  33.7  29.1  37.2     2404
-0.3..-0.1  41.0  28.6  30.4     2280
-0.1..0.1   44.3  28.4  27.4     2574
0.1..0.3    49.9  25.8  24.2     2223
0.3..0.6    57.1  24.8  18.1     2394
0.6..1      65.8  21.4  12.8     1681
> 1         80.1  12.7   7.1     1148
```
