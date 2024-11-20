import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.patches import Circle
import itertools

def retorno_simple(precio_inicial, precio_final):
    """
    Calcula el retorno simple de una inversión basado en el precio inicial y el precio final.

    El retorno simple es una métrica financiera que indica el cambio proporcional en el valor
    de una inversión, expresado como una fracción. Se calcula dividiendo el precio final entre 
    el precio inicial y restando 1.

    Parameters
    ----------
    precio_inicial : int o float
        El precio inicial de la inversión. Debe ser un valor numérico mayor a cero.
    precio_final : int o float
        El precio final de la inversión. Debe ser un valor numérico mayor o igual a cero.

    Returns
    -------
    float
        El retorno simple de la inversión, calculado como una fracción decimal. Por ejemplo,
        un valor de 0.2 indica un retorno del 20%.

    Raises
    ------
    TypeError
        Si `precio_inicial` o `precio_final` no son valores numéricos (int o float).

    Examples
    --------
    >>> retorno_simple(100, 120)
    0.2

    >>> retorno_simple(100, 80)
    -0.2

    """

    # if not isinstance(precio_inicial, (int, float)) or not isinstance(precio_final, (int, float)):
    #     raise TypeError("Los parámetros 'precio_inicial' y/o 'precio_final' deben de ser valores numéricos")
    verificar_valor_numerico(precio_inicial, precio_final)
    retorno = (precio_final / precio_inicial) - 1

    return retorno


def retorno_compuesto(valores):
    """
    Calcula el retorno compuesto de una inversión basado en una lista de valores de precios sucesivos.

    El retorno compuesto es una métrica financiera que mide el crecimiento acumulado de una inversión
    a lo largo de múltiples periodos. Este retorno se calcula multiplicando los retornos de cada
    periodo sucesivo y restando 1 al resultado acumulado.

    Parameters
    ----------
    valores : list of int or float
        Lista de precios sucesivos de la inversión, donde cada elemento representa el valor de la
        inversión en un periodo determinado. La lista debe contener al menos dos valores.

    Returns
    -------
    float
        El retorno compuesto de la inversión, expresado como una fracción decimal. Por ejemplo,
        un valor de 0.3 indica un retorno compuesto del 30%.

    Raises
    ------
    ValueError
        Si la lista `valores` contiene menos de dos elementos.
    TypeError
        Si algún valor en `valores` no es numérico (int o float).

    Examples
    --------
    >>> retorno_compuesto([100, 110, 121])
    0.21

    >>> retorno_compuesto([100, 90, 81])
    -0.19

    """

    if len(valores) < 2:
        raise ValueError('A la lista le faltan más valores')

    # Calcular el retorno de cada periodo y acumular el retorno compuesto
    retorno_compuesto = 1

    for i in range(1, len(valores)):
        precio_inicial = valores[i - 1]
        precio_final = valores[i]

        verificar_valor_numerico(precio_inicial, precio_final)
    
        retorno_periodo = (precio_final / precio_inicial) - 1
        retorno_compuesto *= (1 + retorno_periodo)

    retorno_compuesto = retorno_compuesto - 1

    return retorno_compuesto


def retorno_logaritmico(df):
    df = df.copy()
    df['ret_ln'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

    return df

def retorno_estrategia(df):
    df = df.copy()
    df['ret_estr'] = df['ret_ln'] * df['posicion'].shift(1)
    
    return df
        

def verificar_valor_numerico(valor_inicial, valor_final):
    """
    Verifica que los valores proporcionados sean numéricos.

    Esta función valida que ambos parámetros de entrada, `valor_inicial` y `valor_final`, sean
    de tipo numérico (int o float). Si alguno de los valores no es numérico, lanza una excepción
    `TypeError`.

    Parameters
    ----------
    valor_inicial : int or float
        El valor inicial a verificar. Debe ser numérico.
    valor_final : int or float
        El valor final a verificar. Debe ser numérico.

    Raises
    ------
    TypeError
        Si `valor_inicial` o `valor_final` no son valores numéricos (int o float).

    Examples
    --------
    >>> verificar_valor_numerico(100, 120)
    # No se lanza ninguna excepción

    >>> verificar_valor_numerico(100, "120")
    TypeError: Los parámetros 'precio_inicial' y/o 'precio_final' deben de ser valores numéricos

    """

    if not isinstance(valor_inicial, (int, float)) or not isinstance(valor_final, (int, float)):
            raise TypeError("Los parámetros 'precio_inicial' y/o 'precio_final' deben de ser valores numéricos")


def promedios_moviles(df, promedio_movil1, promedio_movil2):

    df_retornos = np.log(df / df.shift(1))
    df_pm1 = df.rolling(promedio_movil1).mean()
    df_pm2 = df.rolling(promedio_movil2).mean()
    
    df_pm2.dropna(inplace=True)
    
    df = df.loc[df_pm2.index]
    df_retornos = df_retornos.loc[df_pm2.index]
    df_pm1 = df_pm1.loc[df_pm2.index]
    
    df_posicion = pd.DataFrame(np.where(df_pm1 > df_pm2, 1, 0), index=df.index, columns=df.columns)
    df_estrategia = df_retornos * df_posicion.shift(1)    
    
    return df_estrategia.sum().sort_values(ascending=False)



def analizis_promedios_moviles(df, secuencia, titulo=''):
    df_resultados = pd.DataFrame()
    pares = combinar_pares(secuencia)
    
    for par in pares:
        df_resultados[f'{par[0]}-{par[1]}'] = procesar_pares(par, df)

    df_resultados.sort_index(inplace=True)
    df_maximos = mayor_rendimiento(df_resultados)
    
    heatmap = plot_heatmap(df_resultados, titulo)
    columnas = plot_columnas(df_maximos, titulo)

    return heatmap, columnas
    
  
def combinar_pares(secuencia):
    return list(itertools.combinations(secuencia, 2))


def procesar_pares(par, df):   
    return promedios_moviles(df, par[0], par[1]) 


def mayor_rendimiento(df):
    columna_max = df.idxmax(axis=1)
    valor_max = df.max(axis=1)
    maximos = pd.DataFrame({'col_max': columna_max, 'valor_max': valor_max})

    return maximos
    
    
def plot_heatmap(df, titulo):
  fig, ax = plt.subplots(figsize=(14, 8))
  ax.set_title(f'HeatMap: Rendimiento de acciones por pares de promedios móviles {titulo}')
  
  im = ax.imshow(df, cmap='gist_rainbow', aspect='auto')
  fig.colorbar(im, ax=ax)

  # Set x and y labels using tick positions and DataFrame elements
  ax.set_xticks(range(len(df.columns)))
  ax.set_xticklabels(df.columns, rotation=90)
  ax.set_yticks(range(len(df)))
  ax.set_yticklabels(df.index)

  return fig


def plot_columnas(df, titulo):
    contar_promedios = df['col_max'].value_counts()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'Ranking de los pares de promedios móviles {titulo}')

    ax.bar(contar_promedios.index, contar_promedios.values)
    ax.set_xlabel('Promedios móviles')
    ax.set_ylabel('Cantidad')

    return fig


def plot_mercado_estrategia(df):
    fig, ax = plt.subplots(figsize=(16,8))

    ax.plot(df.index, np.exp(df['ret_ln'].cumsum()), label='Retorno mercado')
    ax.plot(df.index, np.exp(df['ret_estr'].cumsum()), label='Retorno estrategia')
    
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Retorno acumulado')
    ax.set_title(f'Estrategia Vs. Mercado')
    ax.legend()
    
    return fig


def balance_monetario(df, saldo_inicial=100):
    return saldo_inicial * (1 + df['ret_estr']).cumprod()


def balances_maximos(serie):
    return serie.cummax()


def plot_perdida_maxima(balance, maximos):
    bajadas = (balance - maximos) / maximos
    mayor_perdida = bajadas.min()
    fecha_mayor_perdida = bajadas.idxmin()
    maximo = maximos[fecha_mayor_perdida]
    balance_perdida = balance[fecha_mayor_perdida]
    perdida_maxima = (maximo - balance_perdida).round(2)

    fig = plt.figure(figsize = (12, 6))
    fig.suptitle(f'Caida máxima de {(mayor_perdida * 100).round(2)}% con fecha de {fecha_mayor_perdida}')
    balance.plot(title = (f'Péridad máxima: ${perdida_maxima}'))
    maximos.plot()
    
    plt.vlines(x=fecha_mayor_perdida, ymin=balance_perdida, ymax=maximo, color='r')
    plt.text(fecha_mayor_perdida, (maximo - balance_perdida*.1).round(2), f' ${perdida_maxima}', color='r')

    return fig


def backtesting(df):
    df = retorno_logaritmico(df)
    df = retorno_estrategia(df)
    balance = balance_monetario(df)
    maximos = balances_maximos(balance)

    grafica_mercado_estrategia = plot_mercado_estrategia(df)
    grafica_perdida_maxima = plot_perdida_maxima(balance, maximos)

    return grafica_mercado_estrategia, grafica_perdida_maxima
    

def analizis_bandas_bollinger(df, ventana, sigma):

    serie_resultados = pd.Series()
    for ticker in list(df.columns):
        datos = pd.DataFrame(df[ticker])
        serie_resultados[ticker] = bandas_bollinger(datos, ventana, sigma)

    serie_resultados.sort_index(inplace=True)

    return serie_resultados
    
    # heatmap = plot_heatmap(df_resultados, titulo)
    # columnas = plot_columnas(df_maximos, titulo)


def bandas_bollinger(df, ventana, sigma):
    df = df.copy()
    df = calculo_banda_media_precio(df, ventana)
    df = calculo_bandas_exteriores(df, sigma)
    df = bandas_bollinger_posicion(df, sigma)
    df = retorno_logaritmico(df)
    df = retorno_estrategia(df)

    return df['ret_estr']


def calculo_banda_media_precio(df, ventana):
    df = df.copy()
    df['bp'] = df['Adj Close'].rolling(ventana).mean() # banda media del precio

    return df


def calculo_bandas_exteriores(df, sigma):
    df = df.copy()
    df[f'+{sigma}sigma'] = df['bp'] + k * df['Adj Close'].rolling(ventana).std() # banda bollinger superior
    df[f'-{sigma}sigma'] = df['bp'] - k * df['Adj Close'].rolling(n).std() # banda bollinget inferior
    df.dropna(inplace=True)

    return df


def bandas_bollinger_posicion(df, k):
    df = df.copy()
    df['posicion'] = np.nan
    df['posicion'] = np.where(df['Adj Close'] < df[f'-{k}sigma'], 1, df['posicion'])
    df['posicion'] = np.where(df['Adj Close'] > df[f'+{k}sigma'], 0, df['posicion'])
    df['posicion'].ffill(inplace=True)
    df.dropna(inplace=True)

    return df


def fig_2activos(df_2activos):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_title('Gráfica Activos Rojo y Azul')

    plt.xlabel('Riesgo', fontsize = 10)
    plt.ylabel('Retorno', fontsize = 10)
    # Limitar los ticks a unicamente los valores de ret y sigma en df_2activos:
    plt.xticks(df_2activos.loc['sigma'])
    plt.yticks(df_2activos.loc['ret'])

    #lista de colores 
    list_color=['red', 'blue']
    i = 0

    """
    For loop para imprimir la ubicación de cada activo, asi como las líneas verticales y horizontales.
    Para que las líneas sobrepasaran el punto le sume + .01 a la longitud de la líneas
    Poner atención como cambia el color de acuerdo al contador 'i':
    """
    for activo in df_2activos.columns:
        ax.hlines(df_2activos[activo].loc['ret'], 0, df_2activos[activo].loc['sigma'] + .01, lw=.8, ls='--', color=list_color[i])
        ax.vlines(df_2activos[activo].loc['sigma'], 0, df_2activos[activo].loc['ret'] + 0.01, lw=.8, ls='--', color=list_color[i])
        plt.scatter(df_2activos[activo].loc['sigma'], df_2activos[activo].loc['ret'], marker = '.', s=1500, label= activo, color=list_color[i], alpha=.15)
        plt.scatter(df_2activos[activo].loc['sigma'], df_2activos[activo].loc['ret'], marker = '.', s=300, label= activo, color=list_color[i])
        i += 1

    #Quitar las líneas del perímetro de la gráfica   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    #Agregar flechas para el eje de las X's y Y's
    arrow_x = FancyArrow(0, 0, .35, 0, head_width=0.005, head_length=0.006, fc='gray', ec='gray', linewidth=0.1)
    arrow_y = FancyArrow(0, 0, 0, 0.15, head_width=0.005, head_length=0.006, fc='gray', ec='gray', linewidth=0.1)
    ax.add_patch(arrow_x)
    ax.add_patch(arrow_y)
    
    return fig


def fig_pregunta(df_2activos):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_title=('sdsd')

    plt.xlabel('Riesgo', fontsize = 14)
    plt.ylabel('Retorno', fontsize = 14)
    # Limitar los ticks a unicamente los valores de ret y sigma en df_2activos:
    plt.xticks(df_2activos.loc['sigma'])
    plt.yticks(df_2activos.loc['ret'])

    #lista de colores 
    list_color=['red', 'blue']
    i = 0

    """
    For loop para imprimir la ubicación de cada activo, asi como las líneas verticales y horizontales.
    Para que las líneas sobrepasaran el punto le sume + .01 a la longitud de la líneas
    Poner atención como cambia el color de acuerdo al contador 'i':
    """
    for activo in df_2activos.columns:
        ax.hlines(df_2activos[activo].loc['ret'], 0, df_2activos[activo].loc['sigma'] + .01, lw=.8, ls='--', color=list_color[i])
        ax.vlines(df_2activos[activo].loc['sigma'], 0, df_2activos[activo].loc['ret'] + 0.01, lw=.8, ls='--', color=list_color[i])
        plt.scatter(df_2activos[activo].loc['sigma'], df_2activos[activo].loc['ret'], marker = '.', s=1500, label= activo, color=list_color[i], alpha=.15)
        plt.scatter(df_2activos[activo].loc['sigma'], df_2activos[activo].loc['ret'], marker = '.', s=300, label= activo, color=list_color[i])
        i += 1

    #Quitar las líneas del perímetro de la gráfica   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    #Agregar flechas para el eje de las X's y Y's
    arrow_x = FancyArrow(0, 0, .35, 0, head_width=0.005, head_length=0.006, fc='gray', ec='gray', linewidth=0.1)
    arrow_y = FancyArrow(0, 0, 0, 0.15, head_width=0.005, head_length=0.006, fc='gray', ec='gray', linewidth=0.1)
    ax.add_patch(arrow_x)
    ax.add_patch(arrow_y)

    ax.scatter(.25, .08, marker = '.', s=1500, label= activo, color='green', alpha=.15)
    ax.scatter(.25, .08, marker = '.', s=300, label= activo, color='green')
    ax.set_xticks([.2,.25,.3])
    ax.set_yticks([.04, .08, .12])
    ax.plot([.2 , .3], [.04, .12], lw=.8, ls='--', color='green')
    ax.hlines(.08, 0, .25 + .01, lw=.8, ls='--', color='green')
    ax.vlines(.25, 0, .08 + 0.01, lw=.8, ls='--', color='green')


def fig_portafolio(df_2activos, rho):

    df_portafolio2activos = portafolio(df_2activos, rho)
    df_portafolio2activos['colores'] = df_portafolio2activos.apply(lambda row: mezcla_colores(row['w_rojo'], 'ff0000', '0018ff'), axis=1)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_title('Portafolios con diferentes proporciones de pesos en los activos "Azul" y "Rojo"; ' + r'$\rho$' + f'={rho}')

    plt.yticks([.04,.12])

    ax.scatter(df_portafolio2activos['sigma'].iloc[1:-1], df_portafolio2activos['retorno'].iloc[1:-1], color=df_portafolio2activos['colores'].iloc[1:-1], s=100)

    colores=['#ff0000', '#0018ff']

    i = 0
    for activo in df_2activos.columns:
        plt.scatter(df_2activos[activo].loc['sigma'], df_2activos[activo].loc['ret'], marker = '.', s=1500, label= activo, color=colores[i], alpha=.15)
        plt.scatter(df_2activos[activo].loc['sigma'], df_2activos[activo].loc['ret'], marker = '.', s=300, label= activo, color=colores[i])
        i += 1

    ax.set_xlabel('sigma')
    ax.set_ylabel('retorno')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False);
    

def fig_port_dif_correlacion(df_2activos):
    rho=.1
    list_color = ['red', 'blue']

    df_portafolio2activos = portafolio(df_2activos, rho)
    list_rho = np.arange(0, 1.01, .1)
    fig, ax = plt.subplots(figsize=(12,6))

    for rho in list_rho:
        df_portafolio2activos = portafolio(df_2activos, rho, False)
        ax.plot(df_portafolio2activos['sigma'], df_portafolio2activos['retorno'], label=f'rho={rho:.1f}')

    i = 0
    for activo in df_2activos.columns:
        plt.scatter(df_2activos[activo].loc['sigma'], df_2activos[activo].loc['ret'], marker = '.', s=1500, color=list_color[i], alpha=.15)
        plt.scatter(df_2activos[activo].loc['sigma'], df_2activos[activo].loc['ret'], marker = '.', s=300, color=list_color[i],)
        i += 1

    ax.set_xlabel('sigma')
    ax.set_ylabel('retorno')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.legend(frameon=False);


def portafolio(df_2activos, r, pocos=True):
    """
    Cálculo vectorizado de los pesos, retornos y volatilidad
    de un portafolio de 2 activos.
    requiere un dataframe con los retornos y volatilidades
    de cada uno de los activos y su correlación
    """
    if pocos == True:
        a = np.arange(0.0, 1.1, .1)
    elif pocos == False:
        a = np.arange(0.0, 1.01, .01)
    df_ = pd.DataFrame()
    df_['w_rojo'] = a
    df_['w_azul'] = 1 - df_['w_rojo']
    df_['retorno'] = (
        df_2activos['rojo'].loc['ret'] * df_['w_rojo'] +
        df_2activos['azul'].loc['ret'] * df_['w_azul']
    )

    df_['sigma'] = (
        (df_2activos['rojo'].loc['sigma']**2)*(df_['w_rojo']**2) +
        (df_2activos['azul'].loc['sigma']**2)*(df_['w_azul']**2) +
        (2 * 
         df_['w_rojo'] * 
         df_['w_azul'] * 
         df_2activos['rojo'].loc['sigma'] * 
         df_2activos['azul'].loc['sigma'] *
         r
        ) 
    )**.5
    
    return df_


def mezcla_colores(peso, color1, color2):
    # Descomponemos los colores en sus componentes R, G, B
    componente1 = tuple(int(color1[i:i+2], 16)/255 for i in (0, 2, 4))
    componente2 = tuple(int(color2[i:i+2], 16)/255 for i in (0, 2, 4))

    # Calculamos la media ponderada de las componentes
    color_mezclado = [(peso*componente1[i] + (1-peso)*componente2[i]) for i in range(3)]
    
    # Convertimos el color mezclado a formato hexadecimal
    return '#{:02x}{:02x}{:02x}'.format(*(int(c*255) for c in color_mezclado))


def fig_port_dif_correlacion(df_2activos):
    rho=.1
    list_color = ['red', 'blue']

    df_portafolio2activos = portafolio(df_2activos, rho)
    list_rho = np.arange(0, 1.01, .1)
    fig, ax = plt.subplots(figsize=(12,6))

    for rho in list_rho:
        df_portafolio2activos = portafolio(df_2activos, rho, False)
        ax.plot(df_portafolio2activos['sigma'], df_portafolio2activos['retorno'], label=f'rho={rho:.1f}')

    i = 0
    for activo in df_2activos.columns:
        plt.scatter(df_2activos[activo].loc['sigma'], df_2activos[activo].loc['ret'], marker = '.', s=1500, color=list_color[i], alpha=.15)
        plt.scatter(df_2activos[activo].loc['sigma'], df_2activos[activo].loc['ret'], marker = '.', s=300, color=list_color[i],)
        i += 1

    ax.set_xlabel('sigma')
    ax.set_ylabel('retorno')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.legend(frameon=False);




def portafolio_simulacion_montecarlo(df, rf, sharp_corte, numero_simulaciones):
    retornos = np.log(df/df.shift(1))
    retornos.dropna(inplace=True)
    retorno_promedio = retornos.mean()
    retornos_anuales = retorno_promedio * 252
    sigma = retornos.std()
    sigma_anual = sigma * np.sqrt(252)
    acciones = list(sigma_anual.index)
    sharp = ((retornos_anuales - rf) / sigma_anual).sort_values(ascending=False)
    acciones_portafolio = sharp[sharp >= sharp_corte]
    nombre_acciones = list(acciones_portafolio.index)
    numero_acciones = len(nombre_acciones)
    matriz_cov = retornos[nombre_acciones].cov()

    simulacion = np.zeros((numero_simulaciones, 4 + numero_acciones))

    for i in range(numero_simulaciones):

        # Calcular los pesos (w) de cada activo 
        w = np.random.random(numero_acciones)
        w = w / w.sum()
    
        retorno_portafolio = np.sum(retornos_anuales[nombre_acciones] * w)
        varianza_portafolio = np.dot(w.T ,np.dot(matriz_cov, w))
        sigma_portafolio = np.sqrt(varianza_portafolio) * np.sqrt(252)
        sharp_portafolio = (retorno_portafolio - rf) / sigma_portafolio
    
        simulacion[i, 0] = i + 1
        simulacion[i, 1] = retorno_portafolio
        simulacion[i, 2] = sigma_portafolio
        simulacion[i, 3] = sharp_portafolio
        for j in range(len(w)):
            simulacion[i, 4 + j] = w[j]

    resultados = pd.DataFrame(simulacion)
    nombre_columnas = ['# simulacion', 'Ret', 'Sigma', 'Sharp'] + nombre_acciones
    resultados.columns = nombre_columnas

    max_sharp_ratio = resultados.iloc[resultados['Sharp'].idxmax()]
    min_vol = resultados.iloc[resultados['Sigma'].idxmin()]

    plt.figure(figsize=(12,6))
    plt.scatter(x=resultados.Sigma.values, y=resultados['Ret'].values, c=resultados.Sharp, cmap='winter', label='simulacion')
    plt.title(f'Portafolios Simulación MonteCarlo {numero_simulaciones:,} simulaciones')
    
    clb = plt.colorbar()
    clb.ax.set_title('SharpRatio')
    
    plt.scatter(max_sharp_ratio.iloc[2], max_sharp_ratio.iloc[1], marker='*', color='gold', s=200, label='Port.Max Sharp Ratio')
    plt.scatter(min_vol.iloc[2], min_vol.iloc[1], marker='*', color='r', s=200, label='Port.Min Volatilidad');

    for accion in nombre_acciones:
        scatter = plt.scatter(sigma_anual[accion], retornos_anuales[accion], marker = '+')
        color = scatter.get_facecolor()[0] 
        plt.text(sigma_anual[accion], retornos_anuales[accion], accion, fontsize=10, color=color);
    











