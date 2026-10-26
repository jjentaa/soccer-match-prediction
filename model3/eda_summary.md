# Model 3 EDA summary

## Tactic data

- snapshots: 1,457 for 288 teams, 2010-2015, one per team per year
- raw scale: 20-80, mean 51.0, std 10.5
- mean |correlation| between tactics: 0.17; largest: defencePressure & defenceTeamWidth = 0.51
- first two principal components explain 47% of the variance
- a team keeps its style from one snapshot to the next 59% of the time

## Share of matches with a style on both sides, by season

```
season
2008/2009    0.000
2009/2010    0.337
2010/2011    0.935
2011/2012    0.956
2012/2013    0.931
2013/2014    0.938
2014/2015    0.931
2015/2016    0.953
```

## Correlation of a team's tactic with points won (training matches)

```
buildUpPlayPassing       -0.061
buildUpPlaySpeed         -0.009
chanceCreationCrossing    0.013
chanceCreationPassing     0.027
defenceAggression         0.029
defenceTeamWidth          0.030
chanceCreationShooting    0.036
defencePressure           0.070
```

## Result rate (%) by PageRank gap (training matches)

```
result         H     D     A
bin                         
< −0.9      23.4  23.5  53.1
−0.9..−0.6  29.7  25.9  44.4
−0.6..−0.3  34.6  28.0  37.4
−0.3..0     41.4  27.5  31.1
0..0.3      50.2  25.7  24.2
0.3..0.6    56.1  24.4  19.5
0.6..0.9    64.1  20.9  15.0
> 0.9       73.4  16.0  10.6
```

## Style similarity graph, England Premier League 2015/2016

- teams: 20
- edges: 44
- density: 0.232
- average_clustering: 0.437
- most_connected: Newcastle United (8), Bournemouth (7), Watford (7)
