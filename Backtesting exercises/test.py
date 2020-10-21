class LRClassifier():
    def __init__(self, PAR = 8, BW = 0.16, RR = 1):
        filepath = "./LRClassifier.h5"  ## Nombre de archivo "H5" adonde guardamos la RN.
        self.BW, self.RR, self.Model = BW, RR, None  ## Por ahora, el modelo está vacío.
        if not os.path.isfile(filepath): self.update(PAR)  ## Si no hay "H5", lo crearemos.
        else: self.Model = keras.models.load_model(filepath)  ## De otro modo, cargar modelo.
        self.PAR = self.Model.layers[0].input_shape[1]  ## "PAR": cantidad de "features".
        self.minRows = self.PAR  ## Los features serán los cambios porcentuales de "Close".
    def call(self, Rows):
        Calc = Rows["Close"].pct_change().dropna()
        Calc = Calc.values.reshape(-1, 1, self.PAR)
        pred = float(self.Model.predict(Calc))  ## Probabilidad de próxima vela alcista.
        Trade = OP = SL = TP = None  ## Default: None.
        if (1 > pred > (1 + self.BW)/2): Trade = +1  ## Comprar
        if (0 < pred < (1 - self.BW)/2): Trade = -1  ## Vender
        if Trade:
            H, L, OP = Rows.iloc[-1, :][["High", "Low", "Close"]].values
            K = Trade*(pred - 1/2) - self.BW/2 ## Distancia entre "pred" y banda mas cercana.
            SL = OP - 100*Trade*(H - L)*K  ## Si compro, el SL queda debajo de OP; y viceversa.
            TP = OP + 100*Trade*(H - L)*K*self.RR  ## Si compro, el SL queda encima de OP, y etc.
        Signal = {"Trade": Trade, "OP": OP, "SL": SL, "TP": TP}
        Indicators = {"Pred": pred}  ## Oscilador. Debe graficarse debajo del gráfico de velas.
        return Indicators, Signal
    @staticmethod  ## Puede usarse fuera de la clase.
    def train_data(PAR, tts = 75/25): ## "tts" -> Train-test split.
        ## Los datos para entrenamiento son tomados previos a los de test.
        Train = yfinance.download(tickers = tickers, interval = interval,
                            end = t_final - dt, start = t_final - tts*dt)
        Train["X(t)"] = Train["Close"].pct_change() ## Cambio porcentual
        ## Corrimientos del cambio porcentual, para autoregresión.
        for n in range(1, PAR): Train[f"X(t-{n})"] = Train["X(t)"].shift(n)
        ## Los "labels" serán "True" cuando el cambio porcentual sea positivo.
        Train["X(t+1)"] = (Train["X(t)"].shift(-1) > 0) ## ...o sea: alcista.
        Train.dropna(axis = "rows", how = "any", inplace = True) ## "Limpieza".
        ## "features" ("x"): solo los corrimientos de cambios porcentuales.
        return { "x": Train.iloc[:, -1 - PAR : -1].values,  ## "values" -> numpy.
                 "y": Train.iloc[:, -1].values }  ## "labels": última columna.
    def update(self, PAR, LR = 0.01, epochs = 5, Train = None):
        if (Train == None): Train = LRClassifier.train_data(PAR)
        self.Model = keras.models.Sequential()
        self.Model.add(keras.layers.Dense(units = 4, activation = "relu", input_shape = (PAR,)))
        self.Model.add(keras.layers.Dense(units = 4, activation = "relu"))
        self.Model.add(keras.layers.Dense(units = 1, activation = "sigmoid"))
        self.Model.compile(loss = "binary_crossentropy", metrics = ["accuracy"],
                           optimizer = keras.optimizers.Adam(learning_rate = LR))
        print(self.Model.summary)
        self.Model.fit(x = Train["x"], y = Train["y"], epochs = epochs)
        self.Model.save("LRClassifier.h5")