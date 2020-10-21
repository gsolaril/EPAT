# Other function packages.
import numpy, pandas, pytz, datetime, matplotlib, matplotlib.pyplot, itertools, ProgBar
import IPython.display, scipy.stats, yfinance, MetaTrader5
# Preset (dark) plot properties.
matplotlib.pyplot.style.use("https://raw.githubusercontent.com/"\
 + "gsolaril/Templates4Coding/master/Python/mplfinance.mplstyle")
pandas.plotting.register_matplotlib_converters()
pandas.options.display.max_columns = None
colors = matplotlib.cm.ScalarMappable()

class Backtest(object):

    max_rows = 5000

    def __init__(self, name = "Strategy"):

        df, pm, self.name = pandas.DataFrame, pandas.MultiIndex, name
        self.Data = df(columns = pm.from_tuples(tuples = [], names = ("Symbol", "Value")))
        self.Active = df(columns = ["Time", "Price", "Sym", "Dir", "Size", "Sink", "SL", "SP"])
        self.Trades = df(columns = pm.from_product(iterables = [("Time", "Price"), ("Open", "Close")]))
        for c in ("Sym", "Dir", "Size", "Sink", "Cause"): self.Trades[c] = None
        self.Specs, self.Stats = df(), df()

    def __repr__(self):

        print(f"Strategy ID: '{self.name}'")
        if not self.Stats.empty: display(self.Stats)
        elif not self.Trades.empty: display(self.Trades)
        else: display(self.Data)
        return

    def _load_data_error(symbols, dates, frame, source):

        sources, dt, array_types = ["MT5", "yfinance"], datetime.timedelta, (set, tuple, list, str)
        assert isinstance(symbols, array_types), "{TYPE} 'symbols' must be a string label or an array of them."
        assert isinstance(frame, str), "{TYPE} 'frame' must be given as a string on one of the MT5 enum labels."
        assert isinstance(dates, (list, tuple, numpy.ndarray)) and isinstance(dates[-1] - dates[0], dt) and \
               (len(dates) == 2) and  "{TYPE} 'dates' must be a list or tuple of only 2 'datetime' objects."
        assert (source in sources), "{COND} 'source' must be between the ones available: '%s'" % sources

    def load_data(self, symbols, dates = None, frame = "M5", source = "yfinance"):

        t, dt = datetime.datetime.now(), datetime.timedelta(days = 15)  
        if (dates == None): dates = (t - dt, t)  ## By default, use last 15 days' data in store.
        Backtest._load_data_error(symbols, dates, frame, source)  ## Check that all inputs are right.
        if isinstance(symbols, str): symbols = {symbols}  ## We'll use set methods: assure 'symbols' is a set.
        self._download_specs(symbols, source)  ## Download symbols' constants and append to "Specs" dataframe.
        self._download_data(symbols, dates, frame, source)  ## Download symbols' market data and add to "Data".
        self.Data.interpolate(inplace = True)  ## Complete NaN values with linearly interpolated data.
        self.Data.dropna(inplace = True, how = "all")  ## Delete rows with any residual NaN value.

    def _download_data(self, symbols, dates, frame, source):

        for symbol in symbols:
            assert isinstance(symbol, str), "{TYPE} each 'symbol' in set must be a string and included in MT5."
            if (source == "MT5") and MetaTrader5.initialize(): ## If MT5 is installed and expressed as source...
                enum = eval(f"MetaTrader5.TIMEFRAME_{frame}") ## Find int value in MT5 module attached to frame.
                data = MetaTrader5.copy_rates_range(symbol, enum, *dates) ## Download market data of the symbol.
                data = pandas.DataFrame(data).iloc[:, 0, 1, 2, 3, 4, 7, 6] ## Get DF of it & drop "tick_volume".
                data["time"] = pandas.to_datetime(data["time"], unit = "s") ## Convert time values to datetime.
                data.set_index(inplace = True, keys = "time", drop = True) ## Set datetime values as index.
            else: # if (source == "yfinance"): ...or any other source that we'd treat as default...
                enum = "".join([_ for _ in frame if _.isdigit()]) \
                     + "".join([x for x in frame if not x.isdigit()]) ## "H4" in MT5 is "4h" in yfinance.
                data = yfinance.download(tickers = symbol, rounding = True,
                       interval = enum.lower(), start = dates[0], end = dates[1]) ## Download market data.
                data = data.iloc[:, [0, 1, 2, 4, 5]]  ;  data["spread"] = 0 ## Drop "Close" column (use "Adj") &
            self._stack(data, symbol, frame) ## Add result to database.     ## create/fill "Spread" column w/0s.

    def _stack(self, data, symbol, frame):

        point = self.Specs[symbol].loc["point"] ## Retrieve how much is a symbol's "point" worth in its price.
        data["spread"] = data["spread"]*point  ## Spreads often come expressed in points. Measure them in price.
        data = data.round(self.Specs[symbol].loc["digits"]) ## Round prices with standard amount of decimals.
        OHLCV = ["Open", "High", "Low", "Close", "Volume"] ## Each price dataset will always have these columns.
        for c, column in enumerate(OHLCV): ## Create a data section in dataframe for each symbol-frame dataset.
            self.Data[(f"{symbol} {frame}", column)] = data.iloc[:, c] ## Add each of the columns to section.
        self.Data[(symbol, "Spread")] = data["spread"] ## Create a signal section in dataframe for each symbol.
        for column in ("LB", "SS", "LS", "SB"): self.Data[(symbol, column)] = None ## Add signal columns.

    def _download_specs(self, symbols, source):
        for symbol in symbols:
            if (source == "MT5") and MetaTrader5.initialize(): ## If MT5 is installed and expressed as source...
                specs = MetaTrader5.symbol_info(symbol)._asdict() ## Retrieve symbol specifications as dict.
                self.Specs[symbol] = specs.values() ## Create a column for every symbol, and add its constants.
                self.Specs.index = specs.keys() ## Use rows as each specification description/tag.
            else: ## Let's "invent" the specifications we will really need.
                self.Specs[symbol] = None
                self.Specs.loc["point", symbol] = 0.01
                self.Specs.loc["digits", symbol] = 2
                self.Specs.loc["trade_tick_size", symbol] = 0.01
                self.Specs.loc["trade_tick_value", symbol] = 1.0
                self.Specs.loc["trade_stops_level", symbol] = 0.01

    def drop_data(self, symbol):

        assert isinstance(symbol, str), "{TYPE} 'symbol' must be given as string, and formerly loaded."
        self.Specs.drop(columns = [symbol], inplace = True)  ## Delete symbol column from specs.
        columns = [c for c in self.Data.columns if symbol in c[0]]  ## Get every column name in Data that is...
        self.Data.drop(columns = columns, inplace = True)         ## ...related to the symbol, & delete them.
        self.Data.dropna(inplace = True, how = "all") ## Delete any residual NaN that appeared in the process.

    def _check_min_frame(self, symbol):

        f_min, n_min = "MN1", 50000  ### Max available frame.
        for column in self.Data.columns:
            header = column[0].split(" ")  ### Symbol name
            if (symbol in column[0]) and (len(header) > 1):
                f_new = header[1]  ### Header suffix (timeframe)
                n_new = eval(f"MetaTrader5.TIMEFRAME_{f_new}")
                if (n_new < n_min): n_min, f_min = n_new, f_new
        return f_min

    def _load_signal_error(self, symbol, dire, trig, rule, shift, strict):

        assert (symbol in self.Data.columns), "{INCL} 'symbol' must be included in the available data."
        assert isinstance(trig, tuple), "{TYPE} 'trig' must be a string label of a block column in data."
        assert (trig in self.Data.columns), "{INCL} 'trig' dataset must be included in the available data."
        dirs = ("LB", "SS", "LS", "SB")
        assert isinstance(dire, str), "{TYPE} 'dire' must be given as a two-letter string: LB/SS/LS/SB."
        assert (dire in dirs), "{INCL} 'dire' must be equal to any of the trading signal types: LB/SS/LS/SB."
        assert isinstance(rule, type(lambda: 1)), "{TYPE}: Rule functions must be 'lambda' or 'def' objects."
        assert isinstance(strict, bool), "{TYPE} 'strict' is a boolean flag. Please, use 'True' or 'False'."
        assert isinstance(shift, int) and (shift >= 0), "{COND} 'shift' must be a positive integer."

    def load_signal(self, symbol, dire, trig, rule, shift = 0, strict = True):

        self._load_signal_error(symbol, dire, trig, rule, shift, strict) ## Check if input arguments are OK.
        dataset = symbol + " " + self._check_min_frame(symbol) ## Get symbol's dataset with most precision.
        signal = self.Data.apply(axis = "columns", func = rule).shift(shift) ## Apply rule function to dataset.
        prices = self.Data[trig].where(cond = signal.astype(bool)) ## If rule applies, get trig value. Else NaN.
        invalids = (prices < self.Data[dataset]["Low"]) | (self.Data[dataset]["High"] < prices) # Beyond range.
        # If strict flag on, signals with invalid triggers are turned to None. Else, trigger replaced by Close.
        prices.mask(inplace = True, cond = invalids, other = None if strict else self.Data[dataset]["Close"])
        self.Data[(symbol, dire)] = prices.round(self.Specs[symbol].loc["digits"]) ## Round to standard form.

    def _check_new_trades(self, time, symbol = None, max_trades = 2, size = None, SL = None, SP = None):

        if (symbol == None): symbol = self.Data.columns[0][0].split(" ")[0] # Default: leftmost instrument.
        if (size == None): size = (lambda R, T: 1) # Default size 1. Both args must be there anyway.
        if (SL == None): SL = (lambda R, T: numpy.nan) # Every symbol should have SL & SP functions.
        if (SP == None): SP = (lambda R, T: numpy.nan) # READ CELL ABOVE FOR ARGUMENT DETAILS.
        if (len(self.Active) >= max_trades): return # If no space for more trades, just quit.
        row = self.Data.loc[time, :] # (1) Get all row/candle price/volume/signals' data for given time value.
        op_L, op_S = row[symbol][["LB", "SS"]].notna() ## (2) Check if any long/short signal.
        if (op_L == op_S): return # (3) Both signals invalid (F, F) or valid but opposite (T, T) -> quit.
        dire = "LB"*op_L + "SS"*op_S # (4) Either one or the other is True, so "dire" will be "LB" XOR "SS".
        price = row[symbol][dire] # (4) Retrieve price value which triggers the signal.
        size = numpy.abs(size(row, self.Trades))
        SL = numpy.abs(SL(row, self.Trades))  # (5) Run size, SL & SP function on available present info at...
        SP = numpy.abs(SP(row, self.Trades))  # ...given time (market & trades' context) and calc their value.
        min_dist = (self.Specs.loc["point"]*(1 + self.Specs.loc["trade_stops_level"])).max() # (6) Stop can't
        SL, SP = numpy.nan if (SL < min_dist) else SL, numpy.nan if (SP < min_dist) else SP # get nearer than 
        price = price + (op_L - op_S)*row[symbol]["Spread"] # (7) Spread to entry price.  # min_dist to price.
        self.Active = self.Active.append(ignore_index = True, other = { "Time": time,
            "Price": price, "Size": size, "Sink": price, "Sym": symbol, "Dir": dire[0],
            "SL": price - (op_L - op_S)*SL, "SP": price + (op_L - op_S)*SP })  ### (8)

    def _check_close(self, trade, time, stop):

        if (stop == None) or self.Active.empty: return(False) ## When there are no active trades to close.
        if isinstance(stop, float) and not(stop > 0): return(False) ## When there's no valid signal.
        active = self.Active.iloc[trade] ## (1) Get the info of the active trade about to be closed.
        price = stop if isinstance(stop, float) else active[stop] ## (2) Get the closing price.
        cause = stop if isinstance(stop, str) else "Signal" ## (2) Possible causes: "SL", "SP" or "Signal".
        new_trade = [[_] for _ in list(active.iloc[:-2])] ## (3) Create list to append into "self.Trades".
        new_trade = new_trade[0] + [[time]] + new_trade[1] + [[price]] + new_trade[2:] + [[cause]] ## (4)
        self.Active.drop(index = trade, inplace = True) ## (5) Delete not-active-anymore trade.
        self.Active = self.Active.reset_index().drop(columns = "index") ## (5) Shift rows below, upwards.
        self.Trades = self.Trades.append(ignore_index = True, other = pandas.DataFrame( \
             data = dict(zip(self.Trades.columns, new_trade))))  ;  return(True) ## (6) Record closed trade.

    def _test_strategy_error(self, max_trades, size, SL, SP, deadline):

        assert isinstance(max_trades, int), "{TYPE} 'max_trades' must be an integer"
        assert (0 < max_trades), "{SIZE} 'max_trades' must be a positive integer of at least 1."
        assert (max_trades < len(self.Data)/2), "{SIZE} shouldn't run more than 1 trade per 2 or 3 data rows."
        assert isinstance(deadline, bool), "{TYPE} 'deadline' activation must be True or False (boolean)"
        ERR = lambda S_: "{TYPE} object stored in '%s' argument must be a function of type lambda or def." % S_
        if (size != None): assert isinstance(size, type(lambda: 1)), ERR("size")  #  TRADE
        if (SL != None): assert isinstance(SL, type(lambda: 1)), ERR("SL")        #  MANAGEMENT
        if (SP != None): assert isinstance(SP, type(lambda: 1)), ERR("SP")        #  FUNCTIONS

    def test_strategy(self, max_trades = 1, size = None, SL = None, SP = None, deadline = True):

        self._test_strategy_error(max_trades, size, SL, SP, deadline)
        self.Active, self.Trades, self.Stats = self.Active[0:0], self.Trades[0:0], self.Stats[0:0]
        Spectrum = itertools.product(self.Data.index, self.Specs.columns) ## Seek all symbols at all times.
        for time, symbol in list(Spectrum): ## For each database symbol in each time value in market data...
            row, trade = self.Data.loc[time, :], 0 ## Get data row at given time. Reset trade counter.
            frame = self._check_min_frame(symbol) ## Find timeframe of the most precise symbol's dataset.
            H, L, close = row[symbol + " " + frame].loc[["High", "Low", "Close"]] ## Get row prices' info.
            while True: ## [B] For trades, no index limit will be used: repeat scan until "break".
                try: active = self.Active.iloc[trade, :] ## Get n'th active trade actual information.
                except: break ## [B] If fails, means no more trades remain ahead. Jump to next time/symbol.
                if (symbol != active["Sym"]): ## If the instrument being analyzed is not the same... 
                    trade = trade + 1; continue  ## ...as the active trade, skip and go to next one.
                ## [C] Always keep the most critical sink: lower price when going long, higher when short.
                if (active["Dir"] == "L") and (L < active["Sink"]): self.Active["Sink"].iloc[trade] = L
                if (active["Dir"] == "S") and (active["Sink"] < H): self.Active["Sink"].iloc[trade] = H
                ## Check if there's an exit signal (long sell or short buy) in present data row.
                signal = row[symbol].loc["LS" if (active["Dir"] == "L") else "SB"]
                ## Stop priority: [D1] SL goes before SP, [D2] close by stop goes before close by signal.
                stop = "SL" if (L <= active["SL"] <= H) else ("SP" if (L <= active["SP"] <= H) else signal)
                if (self.Data.index[-1] == time) and deadline: stop = close ## If backtest ends, close all.
                ## If trade closes, don't change "n_trade": next trade shifts upwards due to row delete.
                if self._check_close(trade = trade, time = time, stop = stop): continue
                trade = trade + 1  ### If trade doesn't close, go to next active trade row.
            for symbol in self.Specs.columns: ## [E] Open new trades if available.
                self._check_new_trades(time, symbol, max_trades, size, SL, SP)

    def test_balance(self, cap, lev, lot = 1, reinv = 0, mglvl = 0.5):

        assert not(self.Trades.empty), "{COND} no trades ever executed (check signal conditions or dataset)"
        assert isinstance(cap + lev + lot + reinv + mglvl, (int, float)), "{TYPE} arguments must be numeric."
        assert (0.01 <= lot <= 999.99), "{SIZE} 'lot' size is worldwide normalized between 0.01 and 999.99."
        assert (0 <= reinv <= 1), "{SIZE} 'reinv' must be between 0-1 (can't invest more than what you've got!)"
        assert (0 <= mglvl <= 1), "{SIZE} 'mglvl' must be 0-1 (critical margin depends on broker tolerance)"
        self._test_balance_diff()
        self._test_balance_profit(cap = cap, lot = lot, reinv = reinv)
        self._test_balance_return(cap = cap)
        self._test_balance_margin(cap = cap, lot = lot, lev = lev)
        self.Trades.fillna(value = 0, inplace = True)
        # "Game over": when margin is less than 50%, it's because funds are null.
        game_over = (self.Trades["Margin"]["Level"] <= mglvl)
        self.Trades.loc[game_over, ["Profit", "Return", "Margin"]] = 0
        self.Trades.loc[game_over, ("Return", "Rel.DD")] = 1
        # Rounding and percentage
        self.Trades["Profit"] = numpy.floor(1e2*self.Trades["Profit"])/1e2
        self.Trades["Margin"] = numpy.floor(1e2*self.Trades["Margin"])/1e2
        self.Trades["Return"] = (numpy.floor(1e4*self.Trades["Return"])/1e2).astype(str) + "%"
        self.Trades[("Margin", "Level")] = (1e2*self.Trades["Margin"]["Level"]).astype(int).astype(str) + "%"

    def _test_balance_diff(self):

        sg = "\u03A3"  ### Capital sigma unicode, representing summation.
        ps = self.Specs.loc["trade_tick_size", list(self.Trades["Sym"])].values  ### (1)
        self.Trades[("Points", " ")] = (1*(self.Trades["Dir"] == "L") - (self.Trades["Dir"] == "S")*1) \
                            * (1/ps) * (self.Trades["Price"]["Close"] - self.Trades["Price"]["Open"])
        self.Trades[("Points", " ")] = numpy.floor(self.Trades["Points"][" "])  ### (2)
        self.Trades[("Points",  sg)] = self.Trades["Points"][" "].cumsum() ### (3)
        self.Trades[("Points", "Sink")] = (1*(self.Trades["Dir"] == "L") - (self.Trades["Dir"] == "S")*1) \
                                          * (self.Trades["Price"]["Open"] - self.Trades["Sink"]) * (1/ps)
        self.Trades[("Points", "Sink")] = numpy.floor(self.Trades["Points"]["Sink"])

    def _test_balance_profit(self, cap, lot, reinv):

        sg = "\u03A3"  ### Capital sigma unicode, representing summation.
        pv = self.Specs.loc["trade_tick_value", list(self.Trades["Sym"])].values
        linear = self.Trades["Points"][" "] * self.Trades["Size"] * pv * lot ### (1)
        increment = (cap + linear.cumsum().shift(1, fill_value = 0)) \
                  / (cap + linear.cumsum().shift(2, fill_value = 0)) ### (2)
        increment[increment < 0] = 0
        self.Trades[("Profit", " ")] = linear * (1 + (increment - 1)*reinv) ### (3)
        net_profit = cap + self.Trades["Profit"][" "].cumsum()
        self.Trades[("Profit", sg)] = net_profit  ### (4)
        self.Trades[("Profit", "Abs.DD")] = net_profit.cummax() - net_profit
        
    def _test_balance_return(self, cap):

        sg = "\u03A3"  ### Capital sigma unicode, representing summation.
        first_return = self.Trades["Profit"][" "].iloc[0] / cap
        self.Trades[("Return", " ")] = self.Trades["Profit"][sg].pct_change().fillna(value = first_return)
        self.Trades[("Return", "Rel.DD")] = 1 - self.Trades["Profit"][sg] / self.Trades["Profit"][sg].cummax()

    def _test_balance_margin(self, cap, lot, lev):

        ps = self.Specs.loc["trade_tick_size", list(self.Trades["Sym"])].values
        pv = self.Specs.loc["trade_tick_value", list(self.Trades["Sym"])].values
        self.Trades[("Margin", " ")] = self.Trades["Price"]["Open"] * self.Trades["Size"] *lot*pv / (ps*lev)
        self._test_balance_margin_summ(cap, lot)        

    def _test_balance_margin_summ(self, cap, lot):

        self.Trades[("Margin", "\u03A3")] = None
        t_op, t_cl = self.Trades["Time"].values.T
        # Intersecting intervals add up their margins.
        for trade in range(len(self.Trades)):
            t_mid = t_op[trade] + (t_cl[trade] - t_op[trade])/2  ### (1)
            intersection = (t_op <= t_mid) * (t_mid <= t_cl)  ### (2)
            summ = self.Trades["Margin"].loc[intersection, " "].sum()  ### (3)
            self.Trades[("Margin", "\u03A3")].iloc[trade] = summ
        pv = self.Specs.loc["trade_tick_value", list(self.Trades["Sym"])].values
        ### Critical equity as capital minus critical active losses, and margin levels as equity/margin.
        Equity = self.Trades["Profit"]["\u03A3"].shift(1).fillna(value = cap)
        Equity = Equity - self.Trades["Points"]["Sink"] * pv * lot
        self.Trades[("Margin", "Level")] = Equity / self.Trades["Margin"]["\u03A3"]

    def test_stats(self):

        assert not(self.Trades.empty), "{COND} no trades ever executed (check signal conditions or dataset)."
        symbols = (["All"] if (len(self.Specs.columns) > 1) else []) + list(self.Specs.columns)
        contexts = ["Points", "Profit", "Return"] ## Measuring units.
        self.Stats = pandas.DataFrame(columns = pandas.MultiIndex.from_product(iterables = (symbols, contexts)))
        for symbol in list(self.Specs.columns):
            Sample = self.Trades.copy()
            game_over = len(self.Data) - 1  ## Start considering all theoretical trades and timeline.
            Sample["Return"] = Sample["Return"].replace({"%": ""}, regex = True).astype(float)/100 ## Returns back to floats.
            if (symbol != "All"): Sample = Sample[Sample["Sym"] == symbol] ## Keep trades associated to the analyzed symbol.
            for context in contexts: ## "Points" consider all theoretical trades independent of funds, being filtered later.
                self._test_stats_main(Sample[context][" "], (symbol, context), game_over)
                if (context == "Points"):  ## Discard trades after "game_over": keep just the ones before the instant...
                    Sample = Sample[Sample["Profit"]["\u03A3"] > 0] ## ...when/if funds (accumulated profit) fell to zero.
                    game_over = self.Data.index.get_loc(Sample["Time"]["Close"].max()) ## Get the row nº of such instant.
            for context in contexts[::-1]: ## For all drawdown ratios, "return" results may be necessary.
                self._test_stats_sharpe(Sample[context][" "], (symbol, context))
                self._test_stats_sortino(Sample[context][" "], (symbol, context))
                self._test_stats_drawdown(Sample[context][" "], (symbol, context))
                self._test_stats_higher(Sample[context][" "], (symbol, context))

    def _test_stats_main(self, sample, column, game_over):

        self.Stats.loc["Signals", column] = self.Data[column[0]][["LB", "SS"]][: game_over].count().sum()
        self.Stats.loc["Trades", column] = len(sample)
        self.Stats.loc["R | HIT", column] = len(sample[sample > 0])/len(sample)
        self.Stats.loc["Max. Profit", column] = sample[sample > 0].max()
        self.Stats.loc["Avg. Profit", column] = sample[sample > 0].mean()
        self.Stats.loc["Max. Loss", column] = -sample[sample < 0].min()
        self.Stats.loc["Avg. Loss", column] = -sample[sample < 0].mean()

    def _test_stats_sharpe(self, sample, column):

        self.Stats.loc["Mean", column] = sample.mean()
        s = "\u03C3\u21F3"  ;  self.Stats.loc[s, column] = sample.std()
        self.Stats.loc["R | Sharpe", column] = self.Stats.loc["Mean", column] / self.Stats.loc[s, column]
        self.Stats.loc["N(-Sharpe)", column] = scipy.stats.norm.cdf(-self.Stats.loc["R | Sharpe", column])

    def _test_stats_sortino(self, sample, column):

        s = "\u03C3\u21E9"   ;   self.Stats.loc[s, column] = sample[sample < sample.mean() / 2].std()
        self.Stats.loc["R | Sortino", column]  = self.Stats.loc["Mean", column] / self.Stats.loc[s, column]

    def _test_stats_drawdown(self, sample, column):

        div = sample.cumsum().cummax()*(column[1] == "Return") ### To drawdown
        CL = (sample.cumsum().cummax() - sample.cumsum()) / (1 + div)
        CL.drop_duplicates(keep = "first", inplace = True)
        self.Stats.loc["Max. DD", column] = CL.max()    ;    self.Stats.loc["Mean DD", column] = CL.mean()
        self.Stats.loc["R | Calmar", column] = self.Stats.loc["Mean", column] / self.Stats.loc["Max. DD", column]
        self.Stats.loc["R | Sterling", column] = self.Stats.loc["Mean", column] / self.Stats.loc["Mean DD", column]

    def _test_stats_higher(self, sample, column):

        self.Stats.loc["Mode", column] = sample.mode()[0]  ;  self.Stats.loc["Median", column] = sample.median()
        self.Stats.loc["Skew", column] = sample.skew()     ;  self.Stats.loc["Kurtosis", column] = sample.kurtosis()

    @staticmethod
    def _5format(ticks = datetime.datetime.now()):

        if isinstance(ticks, datetime.datetime): ticks = [ticks]
        labels = []
        for tick in ticks:
            assert isinstance(tick, datetime.datetime), "{TYPE} 'ticks' must be a list of datetime objects."
            Y, M, D, h, m, s, u = *tick.utctimetuple()[:6], int(tick.microsecond//10000)
            label = "%02dS%02d" % (s, u)
            if (label[2] == "S") and (u == 0): label = "%02dM%02d" % (m, s)
            if (label[2] == "M") and (s == 0): label = "%02dH%02d" % (h, m)
            if (label[2] == "H") and (m == 0): label = "%02dD%02d" % (D, h)
            if (label[2] == "D") and (h == 0): label = "%02d/%02d" % (M, D)
            if (label[2] == "/") and (D == 1): label = "%02dY%02d" % (Y % 100, M)
            if (label[2] == "Y") and (M == 1): label = "Y" + Y
            labels.append(label)
        return labels
    
    def _plot_timeline(self, axes, indexes, intervals = 50):

        ticks = numpy.linspace(start = indexes[0], stop = indexes[-1], num = intervals + 1)
        ticks = ticks.astype(int)
        ticklabels = Backtest._5format(ticks = list(self.Data.index[ticks]))
        axes.set_xticks(ticks = ticks)  ;  axes.minorticks_off()
        axes.set_xticklabels(labels = ticklabels, rotation = 90, fontdict = {"fontsize": 12})

    def _plot_time_error(self, t1, t2):

        L = len(self.Data.index) - 1
        t1 = self.Data.index.get_loc(t1) if isinstance(t1, datetime.datetime) else t1
        t2 = self.Data.index.get_loc(t2) if isinstance(t2, datetime.datetime) else t2
        t1 = int(t1*L) if isinstance(t1, float) and (0 <= t1 < 1) else (0 if (t1 == None) else t1)
        t2 = int(t2*L) if isinstance(t2, float) and (0 < t2 <= 1) else (L if (t2 == None) else t2)
        try: t1, t2 = int(t1), int(t2)
        except: assert isinstance(t1, int) and isinstance(t2, int), "{TYPE} 't1' & 't2' must be either:" \
               + f"\n * (int) '0 < t < {L}' as row numbers.\n * (float) '0 < t < 1' as fractions of timeline." \
               + f"\n * (datetime) time values present in dataset.\n * (None) Default: whole dataset timeline." 
        return numpy.clip(a_min = 0, a = [min(t1, t2), max(t1, t2)], a_max = L).astype(int)

    def plot_chart(self, symbol, t1 = None, t2 = None, signals = False, trades = False, balance = False):

        assert isinstance(trades and signals and balance, bool), "{TYPE} flags must all be boolean."
        assert (symbol in self.Specs.columns), "{INCL} 'symbol' must be between the ones in the dataset."
        t1, t2 = self._plot_time_error(t1, t2)  ### (1) Time formatting to dataframes
        OHLCV = self.Data[symbol + " " + self._check_min_frame(symbol)].iloc[t1 : t2]
        config = { "edgecolor": "white", "alpha": 1, "zorder": 3, "width": 0.5,
                    "linewidth": min(12/numpy.sqrt(t2 - t1), 2) } ### (3)
        Figure, Axes = matplotlib.pyplot.subplots();
        x = numpy.arange(start = t1, stop = t2)
        x_wicks = numpy.repeat(x, repeats = 3) ### (5)
        y_wicks = OHLCV.iloc[:, 1 : 3].assign(NaN = None).values.reshape(-1)
        Axes.plot(x_wicks, y_wicks, color = "white", linewidth = config["linewidth"]); ### (6)
        self._plot_timeline(axes = Axes, indexes = x); ### (6)
        bulls, bears = (OHLCV["Open"] <= OHLCV["Close"]), (OHLCV["Open"] > OHLCV["Close"]) ### (7)
        o, c = OHLCV[bulls]["Open"], OHLCV[bulls]["Close"]
        Axes.bar(x = x[bulls], height = c - o, bottom = o, **config, facecolor = "lightgray");
        o, c = OHLCV[bears]["Open"], OHLCV[bears]["Close"] ### (⇦7⇩)
        Axes.bar(x = x[bears], height = o - c, bottom = c, **config, facecolor = "black");
        Axes.set_ylabel(ylabel = symbol, fontsize = 14, rotation = 90); ### Primary axes.
        if signals: self._plot_signals(t1 = t1, t2 = t2, axes = Axes);
        if trades: self._plot_trades(t1 = t1, t2 = t2, axes = Axes);
        if balance: self.plot_balance(t1 = t1, t2 = t2, axes = Axes);
        return Axes
    
    def _plot_signals(self, t1, t2, axes):

        matplotlib.pyplot.sca(axes);
        symbol = axes.get_ylabel(); ### (1)
        L, S = self.Data[symbol].iloc[t1 : t2, 1 : 3].values.T  ### Retrieve just entry signals.
        config = {"zorder": 4, "marker": "o", "s": 20} ### (2)
        xL, xS = numpy.where(L > 0), numpy.where(S > 0)
        axes.scatter(x = list(xL + t1), y = list(L[xL]), **config, color = "blue");
        axes.scatter(x = list(xS + t1), y = list(S[xS]), **config, color = "red");
    
    def _plot_trades(self, t1, t2, axes):

        assert not(self.Trades.empty), "{COND} 'test_balance' function must be executed first."
        matplotlib.pyplot.sca(axes);
        symbol = axes.get_ylabel() ### (1)
        time_1, time_2 = self.Data.index[[t1, t2]] ### (2)
        Trades = self.Trades[self.Trades["Sym"] == symbol] ### (3)
        Trades = Trades.loc[(time_1 <= Trades["Time"]["Open"])
                         & (Trades["Time"]["Close"] <= time_2), :] ### (3)
        Won = Trades.loc[(Trades["Points"][" "] > 0), ["Time", "Price"]]
        Lost = Trades.loc[(Trades["Points"][" "] < 0), ["Time", "Price"]]
        xW, xL, pW, pL, dt = list(), list(), list(), list(), (t2 - t1)/50
        row = lambda t: self.Data.index.get_loc(t)
        for _1, _2 in Won["Time"].values: xW.insert(-1, [row(_1), row(_2), row(_2) - dt, None])
        for _1, _2 in Lost["Time"].values: xL.insert(-1, [row(_1), row(_2), row(_2) - dt, None])
        for _1, _2 in Won["Price"].values: pW.insert(-1, [_1, _2, _2, None])
        for _1, _2 in Lost["Price"].values: pL.insert(-1, [_1, _2, _2, None])
        axes.plot(numpy.array(xW).reshape(-1), numpy.array(pW).reshape(-1), lw = 1.5, color = "limegreen")
        axes.plot(numpy.array(xL).reshape(-1), numpy.array(pL).reshape(-1), lw = 1.5, color = "orangered")

    def plot_balance(self, t1 = None, t2 = None, axes = None):

        assert not(self.Trades.empty), "{COND} 'test_balance' function must be executed first."
        Profit = pandas.Series(index = self.Data.index)   ;   Margin = Profit.copy()  ;  Margin.iloc[0] = 0
        time_open, time_close = self.Trades["Time"].values.T  ### (2)
        Profit.iloc[0] = (self.Trades["Profit"]["\u03A3"] - self.Trades["Profit"][" "]).iloc[0]  ### (3)
        Profit.loc[time_close] = self.Trades["Profit"]["\u03A3"].values  ### (4)
        Margin.loc[time_open] = self.Trades["Margin"]["\u03A3"].values   ### (5)
        Margin.loc[time_close] = (self.Trades["Margin"]["\u03A3"] - self.Trades["Margin"][" "]).values ### (5)
        Profit.fillna(inplace = True, method = "ffill")  ### (6)
        Margin.fillna(inplace = True, method = "ffill")  ### (6)
        if (axes != None): axes = axes.twinx(); ####### If plotting as separate new figure from main chart...
        else: Figure, axes = matplotlib.pyplot.subplots(); ### (7) Create figure/axes.
        t1, t2 = self._plot_time_error(t1, t2);  ### Format time arguments.
        axes.tick_params(axis = "y", colors = "lime", labelsize = 12)  ;  axes.yaxis.grid(False)
        axes.plot(range(t1, t2), Profit.values[t1 : t2], "--", color = "lime"); ### (7)
        axes.plot(range(t1, t2), Margin.values[t1 : t2],  ":", color = "lime"); ### (7)
        axes.set_ylabel("Profit (--) & Margin (\u00b7\u00b7)", rotation = 90, color = "lime", fontsize = 14);
        axes.set_yticks(ticks = numpy.arange(start = 0, stop = Profit.max(), step = Profit.iloc[0]/4));
        self._plot_timeline(axes = axes, indexes = numpy.arange(t1, t2)); ### & format t's.
        return axes

    def plot_indicator(self, labels, t1 = None, t2 = None, axes = None, **kwargs):

        t1, t2 = self._plot_time_error(t1, t2)  ### Format time arguments.
        if isinstance(labels, str): labels = [labels]
        X, Y_ind = range(t1, t2), self.Data["Indicators"].iloc[t1 : t2]  ### (1)
        Warn = "{INCL} 'labels' must be a string or list of strings already present in 'Indicator' block."
        assert all([(label in Y_ind.columns) for label in labels]), Warn
        if (axes != None): Axes = axes ### (2) Use input axes if specified.
        else: Figure, Axes = matplotlib.pyplot.subplots(); ### (2) Else, create new.
        axes_class = type(matplotlib.pyplot.gca())  ### Needed because axes' data type is too long to write.
        assert isinstance(Axes, axes_class), "{TYPE} 'axes' must be an axes from some already plotted figure."
        self._plot_timeline(axes = Axes, indexes = numpy.arange(t1, t2))  ### (3)
        matplotlib.pyplot.sca(Axes) ### Switch actual axes on, to receive future plots.
        if (len(labels) == 1):  ### (4)
            Axes.plot(X, Y_ind[labels[0]], label = labels[0], **kwargs);
        if (len(labels) == 2):  ### (4)
            Axes.fill_between(X, Y_ind[labels[0]], Y_ind[labels[1]], label = str(labels), **kwargs);
        Axes.legend()  ### To distinguish between different indicators.
        return Axes

    def _plot_histogram_error(self, point, size, profit, ret):

        plots = {"point": point, "size": size, "profit": profit, "ret": ret}
        for plot in plots.keys():  # Check that each plot is activated with a correct word or input value.
            color, colors = plots[plot], list(matplotlib.colors.cnames.keys())
            assert isinstance(color, str) or isinstance(color, bool) or (color == None), "{TYPE} " \
                + f"'{plot}' argument must hold a string (color name) or True (random color) if used."
            plots[plot] = "" if (color == False) or (color == None) else color \
                if (color in colors) else colors[numpy.random.randint(0, 147)]
        n_plots = sum([bool(x) for x in plots.values()])
        assert (n_plots > 0), "{COND} at least one metric from (%s) must be activated" % list(plots.keys())
        return plots, n_plots

    def plot_histogram(self, symbol, t1 = None, t2 = None, point = "green",
                       size = "teal", profit = None, ret = None, bins = 5):
        
        plots, n_plots = self._plot_histogram_error(point, size, profit, ret)
        Figure, Axes = matplotlib.pyplot.subplots(ncols = n_plots, sharey = True); ### (1)
        t1, t2 = self._plot_time_error(t1, t2)  ### Format time arguments.
        time_1, time_2 = self.Data.index[[t1, t2]]  ### Get datetime indexes.
        Trades = self.Trades[self.Trades["Sym"] == symbol]  ### (⇦2⇩)
        Trades = Trades.loc[(time_1 <= Trades["Time"]["Open"]) & (Trades["Time"]["Close"] <= time_2), :]
        n_plots = 0  ### (3)
        if plots["size"]:
            Axes[n_plots].hist(bins = bins, color = plots["size"], x = Trades["Size"]); ### (4)
            Axes[n_plots].set_title("Trade size")    ;    n_plots = n_plots + 1 ### (3)
        if plots["point"]:
            Axes[n_plots].hist(bins = bins, color = plots["point"], x = Trades["Points"][" "]); ### (4)
            Axes[n_plots].set_title("Points per trade")    ;    n_plots = n_plots + 1 ### (3)
        if plots["profit"]:
            Axes[n_plots].hist(bins = bins, color = plots["profit"], x = Trades["Profit"][" "]); ### (4)
            Axes[n_plots].set_title("Profit per trade")    ;    n_plots = n_plots + 1 ### (3)
        if plots["ret"]:
            Returns = [float(x[:-1])/100 for x in self.Trades["Return"][" "]]
            Axes[n_plots].hist(bins = bins, color = plots["ret"], x = Returns); ### (4)
            Axes[n_plots].set_title("Return per trade")    ;    n_plots = n_plots + 1 ### (3)
        return Figure

    def plot_complete(self, symbol, t1 = None, t2 = None, signals = False, trades = False, balance = False):
        
        AxChart = self.plot_chart(symbol, t1, t2, signals, trades, balance)
        AxChart.set_position(pos = [0.05, 0.1, 0.68, 0.8]);
        Figure = AxChart.figure
        Figure.set_figheight(Figure.get_figheight())
        AxTable = Figure.add_axes([0.8, 0, 0.2, 0.9])
        Stats = self.Stats[symbol]
        for m, n in enumerate([0, 1, 10, 1000, 10000]):
            f = lambda x: numpy.round(x*10**(4 - m))/10**(4 - m)
            Stats[2:] = Stats[2:].applymap(lambda x: x if (abs(x) <= n) else f(x))
        Table = AxTable.table(rowLabels = Stats.index, colLabels = Stats.columns, cellText = Stats.values,
                      cellColours = len(Stats.index)*[len(Stats.columns)*["black"]], bbox = [0.35, 0, 1, 1],
                      rowColours = len(Stats.index)*["black"], colColours = len(Stats.columns)*["black"])
        for n in range(len(Table.get_children())): Table.get_children()[n].set_fontsize(12)
        AxTable.set_xticks(ticks = [])  ;  AxTable.set_yticks(ticks = [])
        return Figure

    def clone(self, name = "A certain strategy"):
        
        name = name + ("" if (self.name != name) else "´") ## If name coincides with original, add a tilde.
        clone = Backtest(name = name)
        clone.Data = self.Data.copy() ## Make a copy of market data.
        clone.Data.drop(inplace = True, columns = self.Specs.columns) ## Delete signal columns.
        try: clone.Data.drop(inplace = True, columns = "Indicators")  ## Delete indicator column.
        except: x = 1
        for symbol in self.Specs.columns: ## Create signal columns again, but empty. One block for each symbol.
            clone.Data[(symbol, "Spread")] = self.Data[(symbol, "Spread")].copy() ## This should be identical.
            clone.Data[(symbol, "LB")] = None
            clone.Data[(symbol, "SS")] = None 
            clone.Data[(symbol, "LS")] = None
            clone.Data[(symbol, "SB")] = None
        clone.Specs = self.Specs.copy()
        clone.Active = self.Active.copy()[0:0] ## Copy results' dataframes but without rows: leave...
        clone.Trades = self.Trades.copy()[0:0] ## ...them just as blank, but keep the column labels.
        clone.Stats = self.Stats.copy()[0:0]
        return clone

def optimap(f_xy, X_params, Y_params, heat = True, symm = False):
    
    Stats = ["Trades", "Mean", "R | Sharpe", "R | Sterling"] ## Around which statistical metrics are we going to optimize.
    arrays = (list, tuple, dict, range, numpy.ndarray, pandas.Series) ## Indexable arrays for "X" & "Y" parameter arguments.
    assert isinstance(X_params, arrays) and isinstance(Y_params, arrays), "{TYPE} 'X' & 'Y' params must be indexable arrays."
    X_labels, Y_labels = X_params, Y_params ## "Grid" df's index/column labels may tentatively be the parameters themselves.
    ## However, if "X"/"Y" params' array is a dict, keys will be column labels in Grid. Values will be arguments for "f_xy".
    if isinstance(X_params, dict): X_params, X_labels = list(X_params.values()), list(X_params.keys())
    if isinstance(Y_params, dict): Y_params, Y_labels = list(Y_params.values()), list(Y_params.keys())
    Grid = pandas.DataFrame(index = X_labels, columns = pandas.MultiIndex.from_product(iterables = (Stats, Y_labels)))
    Combs = list(itertools.product(range(len(X_params)), range(len(Y_params)))) ## Indexes of all possible "X" & "Y" pairs.
    ## In cases where "X" & "Y" share exact same indicator nature (e.g.: both SMA), omit repeated/swapped cases. Example:
    if symm: Combs = [(nx, ny) for nx, ny in Combs if (X_params[nx] <= Y_params[ny])] ## "(p1, p2) = (p2, p1)". Keep just one.
    Prog = ProgBar.ProgBar(steps = len(list(Combs))) ## Create a progress bar made with characters.
    ## For given parameters "x" and "y", we will run the exercise, find its stats and create a 2D grid.
    for nx, ny in Combs: ## For every combination of parameters.
        x_param, y_param, x_label, y_label = X_params[nx], Y_params[ny], X_labels[nx], Y_labels[ny]
        try: ## Run the exercise with a function as specified.
            S = f_xy(x_param, y_param).Stats[symbol]["Return"] ## From the backtest results, keep only the returns' ".Stats".
            for stat in Stats: Grid.loc[x_label, (stat, y_label)] = S[stat]
        except: 1 ## When it's impossible to calculate stats (e.g.: no signals/trades), forget about errors.
        Prog.up() ## Increase progress bar.
    Figure, Axes = matplotlib.pyplot.subplots(ncols = len(Stats))
    ## Heatmaps will display the most optimal spots in red.
    Grid.replace(to_replace = [-numpy.inf, numpy.inf], value = numpy.nan, inplace = True)
    for n, stat in enumerate(Stats):   ## Infs ⇧ when denominator is 0.
        lim = max(abs(Grid[stat].min().min()), abs(Grid[stat].max().max()))*1.25  ## Y-axes' max span for lines.
        if heat: Axes[n].contourf(*numpy.meshgrid(X_params, Y_params), Grid[stat].values.T); # Heatmap, 2D.
        else: Grid[stat].plot(ylim = [-lim*(Grid[stat] < 0).any().any(), lim], ## When no negative numbers found...
                              ax = Axes[n], legend = False, linewidth = 2.5);  ## ...lowest y-axis point can be 0.
        Axes[n].set_title(stat, fontweight = "bold")
    if not(heat): Axes[0].legend(fontsize = 13) ## Add legend just to the first line plot.
    matplotlib.pyplot.pause(1e-13) ## This line avoids a (quite loooong) Tkinter warning print.
    return Figure, Grid