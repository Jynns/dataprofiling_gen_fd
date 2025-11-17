
Select table1.OriginAirportID, table1.DestAirportID, table1.UniqueCarrier, table1.CRSDepTime , table1.FlightNum, table2.OriginAirportID, table2.DestAirportID, table2.UniqueCarrier, table2.CRSDepTime, table2.FlightNum  from On_Time_On_Time_Performance_2016_1 as table1
JOIN  On_Time_On_Time_Performance_2016_1 as table2 ON ( table1.OriginAirportID = table2.OriginAirportID 
and table1.UniqueCarrier = table2.UniqueCarrier 
and table1.CRSDepTime = table2.CRSDepTime 
and table1.DestAirportID = table2.DestAirportID 
and table1.FlightNum <> table2.FlightNum ) LIMIT 100;