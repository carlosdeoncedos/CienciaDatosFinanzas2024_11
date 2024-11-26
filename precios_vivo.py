import websocket
import json

def recibir_mensaje(ws, mensaje):
    datos = json.loads(mensaje)
    # print(f"Datos recibidos: {datos}")
    # print(f"Par: {datos['s']} \t Precio: {datos['p']} \t Cantidad: {datos['q']}")
    # print(f"Precio: {datos['p']}")
    print(datos['p'])




def conexion_establecida(ws):
    print("Conexión establecida con Binance")


def conexion_cerrada(ws, close_status_code, close_msg):
    print("Conexión cerrada")


par = "BTCUSDT"
websocket_endpoint = "wss://stream.binance.com:9443"
websocket_trades = f'{websocket_endpoint}/ws/{par.lower()}@trade'

ws = websocket.WebSocketApp(
    websocket_trades,
    on_message=recibir_mensaje,  # Asignar función para manejar mensajes
    on_open=conexion_establecida,  # Asignar función para conexión abierta
    on_close=conexion_cerrada      # Asignar función para cierre de conexión
)

# Ejecutar el WebSocket
ws.run_forever()