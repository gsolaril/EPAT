# -*- coding: utf-8 -*-

import csv
import numpy
import random
import tkinter
import tksheet
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg\
     import FigureCanvasTkAgg as TkPlot


class GUI(tkinter.Tk):

    def __str__(self): return("OBSim v1.1")

    def __init__(self, **kwargs):

        tkinter.Tk.__init__(self, **kwargs)

        # ------------------------------- Parameters:

        self.backcolor = "black"
        self.forecolor = "white"
        self.font = ("Arial", "10")

        self.resol = {"x": 0.8*self.winfo_screenwidth()
                     ,"y": 0.8*self.winfo_screenheight()}
                    
        # ------------------------------- Basics:
        
        self.title("OBSim v1.1")
        self.configure(background = self.backcolor)
        self.attributes("-fullscreen", False)
        self.resizable(False, False)
        self.geometry("%ix%i+0+0" % (self.resol["x"]
                                    ,self.resol["y"]))
        div_x = self.resol["x"]/64
        div_y = self.resol["y"]/28
        
        # ------------------------------- Variables:
        
        self.Ask = tkinter.IntVar()
        self.inputs = {}
        
        # ------------------------------- Radio button frame:
        
        inputs = {
         "Bn_Ask" : ((1, 1), 0, "Sell"),
         "Bn_Bid" : ((1, 2), 1, "Buy")}
        
        for key in inputs.keys():
           self.inputs[key] = {}
           self.inputs[key] = tkinter.Radiobutton(
                   master   = self
                  ,font     = self.font
                  ,bg       = self.backcolor
                  ,fg       = self.forecolor
                  ,value    = inputs[key][1]
                  ,text     = inputs[key][2]
                  ,variable = self.Ask
                  ,selectcolor = self.backcolor
                  ,activebackground = self.backcolor
                  ,highlightcolor   = self.backcolor)
           self.inputs[key].place(x = inputs[key][0][0]*div_x
                                 ,y = inputs[key][0][1]*div_y)
           self.inputs[key].select()
           
        # ------------------------------- Input frame:
                
        inputs = {
         "In_Vol" : ((5,  1), (8,  1), "Volume", "w"),
         "In_Prc" : ((5,  2), (8,  2), "Price",  "w"),
         "In_Peg" : ((12, 1), (12, 2), "Pegg", "center"),
         "In_Dsc" : ((16, 1), (16, 2), "Disc", "center"),
         "In_Stp" : ((20, 1), (20, 2), "Stop", "center")}
        
        self.Values = dict(zip(inputs.keys(), [0]*len(inputs.keys())))
        
        for key in inputs.keys():
            self.inputs[key] = {}
            self.inputs[key]["Input"] = tkinter.Entry(
                               master = self
                              ,width  = int(div_x/2))
            self.inputs[key]["Input"].place(
                               x      = inputs[key][0][0]*div_x
                              ,y      = inputs[key][0][1]*div_y)
            self.inputs[key]["Label"] = tkinter.Label(
                               master = self
                              ,font   = self.font
                              ,bg     = self.backcolor
                              ,fg     = self.forecolor
                              ,anchor = inputs[key][3]
                              ,text   = inputs[key][2])
            self.inputs[key]["Label"].place(x = inputs[key][1][0]*div_x
                                           ,y = inputs[key][1][1]*div_y)
        
        # ------------------------------- Update:
        
        self.button = tkinter.Button(
                      master  = self
                     ,font    = self.font
                     ,text    = "Update"
                     ,command = self.Update)
        self.button.place(x   = int(24*div_x)
                         ,y   = int(1.5*div_y))

        # ------------------------------- Order book frame:
        
        start_csv = "B:\Stack\Trading\EPAT\OBsim\Start.csv"
        start_csv = [i for i in csv.reader(open(start_csv), delimiter = ";")]
        start_csv = [numpy.array(start_csv)[:, 0: 3],  # Bid
                     numpy.array(start_csv)[:, 3: 6]]  # Ask
        self.LOB = [
            self.Table(x = 30, y = 1, w = 16, h = 12
                      ,indexes = ["L%d" % L for L in range(11)]
                      ,headers = ["Iceberg", "Volume", "Bid"]
                      ,data    = list(start_csv[0])),
            self.Table(x = 46, y = 1, w = 16, h = 12
                      ,indexes = ["L%d" % L for L in range(11)]
                      ,headers = ["Ask", "Volume", "Iceberg"]
                      ,data    = list(start_csv[1])) ]
        self.LOG = \
             self.Table(x = 30, y = 14, w = 32, h = 13
                        ,indexes = ["%04d" % (L + 1) for L in range(0)]
                        ,headers = ["Volume", "Price", "Pegg"
                                   , "Disc", "Level", "Type"])

        # ------------------------------- Chart frame:

        a = [int(start_csv[1][0, 0])]
        b = [int(start_csv[0][0, 2])]
        s = a[0] - b[0]
        for t in range(99):
            an = a[t]*numpy.exp(random.gauss(0, 1/a[0]))
            sn = s*numpy.exp(random.gauss(0, 0.1))
            a.append(an)
            b.append(an - sn)
        self.Figure = Figure()
        self.Figure.set_facecolor(color = self.backcolor)
        self.Figure.set_figwidth(val = 4.6)
        self.Figure.set_figheight(val = 4.8)
        self.Plot = self.Figure.add_subplot(111)
        self.Plot.set_facecolor(color = self.backcolor)
        self.Plot.tick_params(colors = self.forecolor)
        xmax, ymax, ymin = 2*len(a), max(a), 2*b[-1] - max(a)
        self.Plot.hlines(y = 0, xmin = 0, xmax = 200, colors = self.forecolor)
        self.Plot.vlines(x = 0, ymin = 0, ymax = 200, colors = self.forecolor)
        self.Plot.set_xlim(xmin = 0, xmax = xmax)
        self.Plot.set_ylim(ymin = ymin, ymax = ymax)
        self.Figure.tight_layout()
        self.APlot = self.Plot.plot(range(100), a[:: -1], color = "blue")
        self.BPlot = self.Plot.plot(range(100), b[:: -1], color = "red")
        self.Chart = TkPlot(self.Figure, master = self)
        self.Chart.get_tk_widget().place(x = 0.5*div_x, y = 4*div_y)
        self.Chart.draw()
        
    # ------------------------------- Update function:
    def Update(self):
        # If zero volume, don't carry on.
        if not (self.inputs["In_Vol"]["Input"].get().isdigit()): return
        # Retrieve each input value. Set to zero if not numeric.
        for key in list(self.Values.keys())[: -1]:
            Input = self.inputs[key]["Input"].get()
            self.Values[key] = int(Input) if Input.isdigit() else 0
        # Get price relative difference.
        diff_price = self.price_rel_info()["p_diff"]
        # Order submission into history log.
        self.Order_Fill()
        # Check order type in terms of price difference.
        if (diff_price <= 0): self.Order_Market()
        else: self.Order_Limit()
        # Plot new values
        self.Order_Draw()
        return

    # ------------------------------- Get price diff., and spreadsheet columns:
    def price_rel_info(self):
        # See if best price is best ask or best bid based in op-type:
        p_col = 2*(1 - self.Ask.get())
        # Retrieve best price as level 1 (row = 0) of the spreadsheet.
        best_price  = int(self.LOB[self.Ask.get()].get_cell_data(0, p_col))
        best_volume = int(self.LOB[self.Ask.get()].get_cell_data(0, 1))
        # Retrieve worst price as last filled price of the spreadsheet.
        w = len(self.LOB[self.Ask.get()].get_column_data(p_col)) - 1
        worst_price = int(self.LOB[self.Ask.get()].get_cell_data(w, p_col))
        # Market orders:  "diff_price = 0"
        if (self.Values["In_Prc"] == 0): self.Values["In_Prc"] = worst_price
        # Limit orders:   "diff_price > 0"
        diff_volume = (self.Values["In_Vol"] - best_volume)
        diff_price  = (self.Values["In_Prc"] - best_price)*(-1)**self.Ask.get()
        # Return price relative diff., and price/column spreadsheet column.
        # print("%s | order_price = %d | diff_price = %d | diff_volume = %d"
        #  % (self.Ask.get(), self.Values["In_Prc"], diff_price, diff_volume))
        return({"p_col": p_col, "p_diff": diff_price, "v_diff": diff_volume})

    # ------------------------------- Order submission:
    def Order_Fill(self):
        # Add row at the end of order log, with new values.
        self.LOG.insert_row(values = list(self.Values.values()))
        # Write op-type, as it is on a separate property.
        r = self.LOG.total_rows() - 1
        c = self.LOG.total_columns() - 1
        o = "Buy" if self.Ask.get() else "Sell"
        self.LOG.set_cell_data(r, c, o)
        # Show changes.
        self.LOG.refresh()
        return

    # ------------------------------- Market order writer: MONKA
    def Order_Market(self):
        # Get price relative info.
        diff_volume = self.price_rel_info()["v_diff"]
        # In case there's more offer than demand, order is fully executed.
        if (diff_volume < 0):
            # Leave 1st level price. Only reduce offer volume.
            diff_volume = abs(diff_volume)
            self.LOB[self.Ask.get()].set_cell_data(0, 1, str(diff_volume))
            self.LOB[self.Ask.get()].refresh()
            return
        # Order partially executed. Evaluate what's left.
        else:
            self.Values["In_Vol"] = abs(diff_volume)
            # Delete row of recently consumed level.
            self.LOB[self.Ask.get()].delete_row(0)
            self.LOB[self.Ask.get()].refresh()
            # Retrieve updated relative price difference.
            diff_price  = self.price_rel_info()["p_diff"]
            if (diff_price <= 0): self.Order_Market()
            else: self.Order_Market()
        return

    # ------------------------------- Limit order writer: MONKA
    def Order_Limit(self):
        # Get price info:
        p_col = 2 - self.price_rel_info()["p_col"]
        prices = self.LOB[not(self.Ask.get())].get_column_data(p_col)
        prices = list(map(int, prices))
        self.Values["In_Prc"] = self.Values["In_Prc"]
        prices.append(self.Values["In_Prc"])
        prices = sorted(prices, reverse = self.Ask.get())
        lvl = prices.index(self.Values["In_Prc"])
        v = [0, self.Values["In_Vol"], self.Values["In_Prc"]]
        v = sorted(v, reverse = not(self.Ask.get()))
        if (prices.count(self.Values["In_Prc"]) <= 1):
            self.LOB[not(self.Ask.get())].insert_row(idx = lvl, values = v)
            self.LOB[not(self.Ask.get())].refresh()
        else:
            cell = self.LOB[not(self.Ask.get())].get_cell_data(lvl, 1)
            cell = str(self.Values["In_Vol"] + int(cell))
            self.LOB[not(self.Ask.get())].set_cell_data(lvl, 1, cell)
            self.LOB[not(self.Ask.get())].refresh()
        return

    # ------------------------------- Draw new price and volume values:
    def Order_Draw(self):
        a = list(self.APlot[0].get_ydata())
        b = list(self.BPlot[0].get_ydata())
        a.append(int(self.LOB[1].get_cell_data(0, 0)))
        b.append(int(self.LOB[0].get_cell_data(0, 2)))
        self.APlot[0].set_data(range(len(a)), a)
        self.BPlot[0].set_data(range(len(b)), b)
        self.Chart.draw()
        self.Chart.flush_events()
        return

    # ------------------------------- Spreadsheet init: MONKA
    def Table(self, x = 0, y = 0, w = 15, h = 15,
              headers = [], indexes = [], data = []):
        div_x = self.resol["x"]/64
        div_y = self.resol["y"]/28
        Table = tksheet.Sheet(self
            ,row_index_foreground  = self.forecolor
            ,row_index_background  = "#222222"
            ,header_background     = "#222222"
            ,header_foreground     = self.forecolor
            ,table_background      = self.backcolor
            ,text_color            = self.forecolor
            ,grid_color            = self.backcolor
            ,outline_color         = self.forecolor
            ,outline_thickness     = 1
            ,show_x_scrollbar      = False
            ,show_y_scrollbar      = True
            ,header_font           = ("Arial", 8, "bold", "underline")
            ,font                  = ("Arial", 8, "normal")
            ,data                  = data
            ,headers               = headers
            ,row_index             = indexes
            ,row_index_width       = 0.6*w*div_x/(len(headers) + 0.4)
            ,column_width          = 0.8*w*div_x/(len(headers) + 0.4)
            ,width                 = w*div_x
            ,height                = h*div_y)
        Table.place(x = x*div_x, y = y*div_y)
        return(Table)
       
if (__name__ == "__main__"):
    try:
        OBSim = GUI().mainloop()
    finally: del OBSim