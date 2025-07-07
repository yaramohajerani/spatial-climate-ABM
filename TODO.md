# TODOs

## bugs
- [x] fix production function such that input requirements are informed by network topology 

## immediate 
- [x] Update network topology so firms can also buy form other firms. 
- [x] Figure out why firms aren't getting money from selling to households (household wealth keeps going up and firm wealth keeps going down due to labor transaction, but no effect from sales of goods.)
- [x] Make sure randomized runs can be duplicated exactly with and without climate for comparison
- [x] Create a dictionary of firm input/outputs and locations. Flooding is very spatially heterogenous, so randomized placing is very unlikely to get significant flood risk. 

## longer-term / tentative
- [x] Add capital requirement for production
- [x] Use output of damage functions to proportionally affect productivity, capital, and inventory of firms 
- [ ] Make firm damage functions a function of the sector
- [ ] Add bankrupcy when firms can't pay their fixed costs. Update network topology when new firms pop up.
- [ ] Add damage threshold for households, above which they migrate
- [ ] Add support for netCDF input files
- [ ] Add support for more hazards including wildfires, tropical cyclons, drought
- [ ] Add support for chronic risks