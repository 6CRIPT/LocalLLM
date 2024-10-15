import sys
import os
from llama_cpp import Llama

class LLM():
    ruta_modelo = "llama-2-7b.Q5_K_M.gguf"

    def __init__(self):
        self.modelo = self.cargar_modelo()

    def cargar_modelo(self):
        try:
            llama = Llama(model_path=self.ruta_modelo)
            return llama
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}", file=sys.stderr)
            return None

    def obtener_respuesta(self, prompt):
        if not self.modelo:
            print("El modelo no se pudo cargar.")
            return None

        try:
            respuesta = self.modelo(
                prompt,
                max_tokens=150,  
                temperature=0.1,  
                top_p=0.95,  
                top_k=25
            )
            return respuesta['choices'][0]['text']
        except Exception as e:
            print(f"Error al obtener la respuesta del modelo: {str(e)}", file=sys.stderr)
            return None

def main():
    llm = LLM()
    os.system('cls' if os.name == 'nt' else 'clear')

    while True:
        prompt = input("Introduce tu prompt (escribe 'quit' para salir) -> ")

        if prompt.lower() == "quit":
            print("Saliendo del programa...")
            break

        respuesta = llm.obtener_respuesta(prompt)

        if respuesta:
            print(f"Respuesta generada -> {respuesta}")
        else:
            print("No se pudo generar una respuesta.")
    
if __name__ == "__main__":
    main()