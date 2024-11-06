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
    
        

def verificar_valor_numerico(valor_inicial, valor_final):
    if not isinstance(valor_inicial, (int, float)) or not isinstance(valor_final, (int, float)):
            raise TypeError("Los parámetros 'precio_inicial' y/o 'precio_final' deben de ser valores numéricos")
    











































