from binance.client import Client
from dotenv import load_dotenv
import pandas as pd
import time
import os

load_dotenv() # Cargar las variables del archivo .env

# Inicializar cliente de Binance
clave_api = os.getenv('BINANCE_PUBLIC_EGADE')
clave_secreta = os.getenv('BINANCE_PRIVATE_EGADE')
cliente = Client(clave_api, clave_secreta)

# Variable GLOBAL para seguimiento de las perdidas acumuladas:
perdidas_acumuladas = 0.0


def obtener_tamano_lote(base: str):
    """
    Obtiene la cantidad mínima y el tamaño de paso (LOT_SIZE) para una criptomoneda base.

    Args:
        base (str): Nombre de la criptomoneda base (e.g., 'BTC', 'DOGE').

    Returns:
        tuple[float, float]: Una tupla con:
            - cantidad_minima (float): Cantidad mínima permitida para la cripto base.
            - step_size (float): Incremento permitido para la cantidad.
        Si no se encuentra información, devuelve (None, None).

    Ejemplo:
        obtener_tamano_lote('BTC')
        -> (0.00001, 0.00001) para BTC.
    """
    # Obtener la información general del exchange
    exchange_info = cliente.get_exchange_info()

    # Buscar un par que tenga la cripto como base
    for symbol in exchange_info['symbols']:
        if symbol['baseAsset'] == base:
            # Buscar el filtro LOT_SIZE
            for filtro in symbol['filters']:
                if filtro['filterType'] == 'LOT_SIZE':
                    cantidad_minima = float(filtro['minQty'])
                    step_size = float(filtro['stepSize'])
                    return cantidad_minima, step_size

    # Si no se encuentra ningún par con la cripto base
    print(f"Error: No se encontró información para la criptomoneda base {base}")
    return None, None


def ajustar_cantidad_lot_size(cantidad: float, cantidad_minima: float, step_size: float):
    """
    Ajusta una cantidad para que cumpla con las restricciones del lot size en Binance.

    Args:
        cantidad (float): La cantidad inicial que se desea ajustar.
        min_qty (float): La cantidad mínima permitida para operar el activo.
        step_size (float): El tamaño del paso permitido para el lot size del activo.

    Returns:
        float: La cantidad ajustada que cumple con las restricciones del lot size,
        redondeada a 8 decimales. Si la cantidad es menor que `cantidad_minima`, regresa `0.0`.

    Nota:
        - Si `cantidad` es menor que `cantidad_minima`, no es suficiente para realizar un trade y se regresa `0.0`.
        - La cantidad ajustada se calcula eliminando cualquier sobrante que no sea múltiplo de `step_size`.

    Ejemplo:
        Si `cantidad=0.12345678`, `cantidad_minima=0.01`, y `step_size=0.001`, la función regresará `0.123`.
    """
    
    if cantidad < cantidad_minima:
        return 0.0  # Regresa 0 si la cantidad no es suficiente para realizar un trade
    # Restar el sobrante a la cantidad y redondear a 8 decimales
    
    return round(cantidad - (cantidad % step_size), 8)


    

def obtener_precios_recientes(par: str, limite: int = 10):
    """
    Obtiene los precios de cierre recientes para un par de trading específico en Binance.

    Args:
        par (str): El símbolo del par de trading, por ejemplo, 'BTCUSDT'.
        limite (int, opcional): El número máximo de precios de cierre a obtener. 
            Por defecto, es 10.

    Returns:
        list[float]: Una lista de precios de cierre recientes como valores flotantes.

    Nota:
        - Utiliza el método `get_klines` de la API de Binance para obtener velas (candlesticks) 
          en el intervalo de 1 minuto.
        - Extrae el precio de cierre de cada vela, que corresponde al índice 4 en el array de datos.
    """
    
    velas = cliente.get_klines(symbol=par, interval='1m', limit=limite)
    precios_cierre = []
    for vela in velas:
        precios_cierre.append(float(vela[4]))  # Extraer precios de cierre de cada kline
        
    return precios_cierre


def calcular_momentum(precios):
    """
    Calcula el momentum de una serie de precios.

    El momentum se define como la diferencia entre el último precio y el primer precio 
    de la serie proporcionada.

    Args:
        precios (list[float]): Una lista de precios, ordenada cronológicamente, donde
            el primer elemento representa el precio más antiguo y el último elemento 
            representa el precio más reciente.

    Returns:
        float: El valor del momentum, calculado como `precios[-1] - precios[0]`.

    Nota:
        - Un valor positivo indica que el precio ha aumentado desde el inicio hasta el final.
        - Un valor negativo indica que el precio ha disminuido.
    """
    
    return precios[-1] - precios[0]


def obtener_saldos(par: str):
    """
    Obtiene los saldos disponibles de USDT y del activo base de un par de trading específico.

    Args:
        par (str): El símbolo del par de trading, por ejemplo, 'BTCUSDT'.
            Debe estar en el formato `<ACTIVO_BASE>USDT`.

    Returns:
        tuple[float, float]: Una tupla con dos valores:
            - `saldo_usdt` (float): El saldo disponible de USDT.
            - `saldo_base` (float): El saldo disponible del activo base extraído del par.

    Nota:
        - La función asume que el par termina en 'USDT' para extraer el activo base.
        - Utiliza la API de Binance para consultar los saldos disponibles de los activos.
    """
    
    activo_base = par.replace('USDT', '')  # Extraer el activo base (e.g., BTC)
    saldo_usdt = float(cliente.get_asset_balance(asset='USDT')['free'])
    saldo_base = float(cliente.get_asset_balance(asset=activo_base)['free'])
    
    return saldo_usdt, saldo_base, activo_base


def estrategia_momentum(par: str, treshold: float = 0.5):
    """
    Ejecuta una estrategia de trading basada en el momentum de precios para un par de trading específico.

    La estrategia evalúa el momentum del precio para decidir si comprar, vender o mantener la posición.
    Realiza operaciones de compra o venta en Binance basándose en el threshold (`treshold`) y 
    en los saldos disponibles de USDT y del activo base.

    Args:
        par (str): El símbolo del par de trading, por ejemplo, 'BTCUSDT'.
        treshold (float, opcional): Umbral para el momentum. Si el momentum supera este valor positivo,
            se genera una señal de compra. Si es menor que el valor negativo, se genera una señal de venta.
            El valor predeterminado es 0.5.

    Returns:
        None

    Nota:
        - **Compra:** Se ejecuta si el momentum es mayor que el `treshold` y el saldo de USDT disponible es suficiente.
        - **Venta:** Se ejecuta si el momentum es menor que `-treshold` y el saldo del activo base es suficiente.
        - Si no se cumple ninguna condición, no se realiza ninguna operación.
        - Se ajusta el saldo del activo base para cumplir con las restricciones de tamaño de lote de Binance.

    Estructura del Proceso:
        1. Obtiene los precios recientes y calcula el momentum usando las funciones `obtener_precios_recientes` 
           y `calcular_momentum`.
        2. Consulta los saldos disponibles de USDT y el activo base con la función `obtener_saldos`.
        3. Realiza las siguientes operaciones basándose en el momentum:
            - **Compra:** Orden de compra a mercado con el saldo disponible de USDT.
            - **Venta:** Orden de venta a mercado ajustada al tamaño mínimo de lote (`LOT_SIZE`).
        4. Registra y actualiza las pérdidas acumuladas en las ventas.

    Ejemplo:
        >>> estrategia_momentum('BTCUSDT', treshold=0.5)
        Obteniendo precios para BTCUSDT...
        Momentum: 0.8
        Saldo USDT: 100.0, Saldo BTC: 0.01
        Se detectó una señal de compra.
        Comprado BTCUSDT con 100.0 USDT a 25000.0.

    Excepciones manejadas:
        - Si no se pueden obtener los valores de `LOT_SIZE` para un activo, la operación se cancela.
        - Si el saldo ajustado no cumple con el mínimo permitido, la operación también se cancela.

    """
    
    print(f"Obteniendo precios para {par}...")
    precios = obtener_precios_recientes(par)
    momentum = calcular_momentum(precios)
    print(f"Momentum: {momentum}")

    # Obtener saldos
    saldo_usdt, saldo_base, activo_base = obtener_saldos(par)
    print(f"Saldo USDT: {saldo_usdt}, Saldo {activo_base}: {saldo_base}")

    # Lógica basada en momentum
    if momentum > treshold and saldo_usdt > 10:  # Señal de compra
        print("Se detectó una señal de compra.")
        compra = cliente.order_market_buy(symbol=par, quoteOrderQty=saldo_usdt)
        precio_compra = float(compra['fills'][0]['price'])
        print(f"Comprado {par} con {saldo_usdt} USDT a {precio_compra}.")
        
    elif momentum < -treshold and saldo_base > 0:  # Señal de venta
        global perdidas_acumuladas
        
        print("Se detectó una señal de venta.")

        cantidad_minima, step_size = obtener_tamano_lote(par.replace('USDT', ''))
        if cantidad_minima is None or step_size is None:
            print(f"No se pudo obtener el tamaño de lote para {par}. Operación cancelada.")
            return
    
        # Ajustar balance para cumplir con LOT_SIZE
        saldo_base_ajustado = max(cantidad_minima, saldo_base - (saldo_base % step_size))
    
        if saldo_base_ajustado < cantidad_minima:
            print(f"Saldo ajustado ({saldo_base_ajustado}) es menor que la cantidad mínima ({cantidad_minima}). Operación cancelada.")
            return
    
        # Vender a mercado
        venta = cliente.order_market_sell(symbol=par, quantity=saldo_base_ajustado)
        precio_venta = float(venta['fills'][0]['price'])
        monto_vendido = precio_venta * saldo_base_ajustado
    
        # Actualizar perdidas acumuladas
        perdida_actual = saldo_usdt - monto_vendido
        perdidas_acumuladas += max(0, -perdida_actual)
    
        # Resultados
        print(f"Vendido {saldo_base_ajustado} {par.replace('USDT', '')} al precio {precio_venta}.")
        print(f"Pérdida acumulada actual: {perdidas_acumuladas} USDT.")
    
    else:
        print("No se detectó un momentum significativo. Manteniendo posición.")


def liquidar_todo_a_usdt(par: str):
    """
    Liquida todos los activos restantes, no denominados en USDT, vendiéndolos a mercado para convertirlos en USDT.

    Args:
        par (str): El símbolo del par de trading, por ejemplo, 'BTCUSDT'.
            Debe estar en el formato `<ACTIVO_BASE>USDT`.

    Returns:
        None

    Nota:
        - Si el saldo del activo base (por ejemplo, BTC) es mayor a cero, se realiza una orden de venta a mercado.
        - Utiliza la API de Binance para obtener el saldo actual del activo base y para ejecutar la orden de venta.
        - No realiza ninguna acción si el saldo del activo base es igual a 0.

    Ejemplo:
        >>> liquidar_todo_a_usdt('BTCUSDT')
        Liquidando todo a USDT...
        Vendido 0.5 BTC. Todo ha sido liquidado a USDT.
    """
    
    print("Liquidando todo a USDT...")
    activo_base = par.replace('USDT', '')
    _, saldo_base = obtener_saldos(par)
    if saldo_base > 0:
        cliente.order_market_sell(symbol=par, quantity=saldo_base)
        print(f"Vendido {saldo_base} {activo_base}. Todo ha sido liquidado a USDT.")


# Ejecución principal
if __name__ == "__main__":
    
    print(f"Iniciando mi primer TRADING ALGORITMICO!!!")
    par_trading = input("Ingrese el par de trading (e.g., BTCUSDT): ").upper()
    treshold = float(input("Ingrese el umbral de momentum (e.g., 0.5): "))
    try:
        perdida_maxima_permitida = float(input("Ingrese la pérdida máxima permitida en USDT (e.g., 10.0): ") or 10.0)
    except ValueError:
        perdida_maxima_permitida = 10.0  # Default 10.0 si es un input invalido
        
    while True:
        try:
            # Verifica si sobrepasamos la perdida máxima
            if perdidas_acumuladas > perdida_maxima_permitida:
                print(f"Pérdidas acumuladas ({perdidas_acumuladas} USDT) han superado el límite permitido ({perdida_maxima_permitida} USDT).")
                liquidar_todo_a_usdt(par_trading)
                print("Estrategia detenida.")
                break  # Salir del Loop
            
            # Ejecuta la estrategia de trading
            estrategia_momentum(par_trading, treshold)
            time.sleep(10)  # Espera 10 segundos entre análisis
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)
