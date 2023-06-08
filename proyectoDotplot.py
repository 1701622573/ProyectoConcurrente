import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
from Bio import SeqIO
from mpi4py import MPI
import argparse
from tqdm import tqdm
import numba as nb

def merge_sequences_from_fasta(file_path):
  sequences = []
  for record in SeqIO.parse(file_path, "fasta"):
      sequences.append(str(record.seq))
  return "".join(sequences)

def cargar_secuencia(fasta_file1,fasta_file2, porc):
  merged_sequence_1 = merge_sequences_from_fasta(fasta_file1)
  merged_sequence_2 = merge_sequences_from_fasta(fasta_file2)
  print(f"longitud {fasta_file1} 1:", len(merged_sequence_1), fasta_file1)
  print(f"longitud {fasta_file2} 2:", len(merged_sequence_2), fasta_file2)
  numPerc = porc / 100
  cantSeq = int(len(merged_sequence_1) * (numPerc))
  #print(cantSeq)
  Secuencia1Completa = " "
  Secuencia2Completa = " "
  for record in SeqIO.parse(fasta_file1, "fasta"):
    Secuencia1Completa = str(record.seq)[:cantSeq]
  for record in SeqIO.parse(fasta_file2, "fasta"):
    Secuencia2Completa = str(record.seq)[:cantSeq]
  print("secuencia1:",Secuencia1Completa)
  print("longitud secuencia1",len(Secuencia1Completa))
  print("secuencia2:",Secuencia2Completa)
  print("longitud secuencia2",len(Secuencia2Completa))
  Secuencia1 = Secuencia1Completa
  Secuencia2 = Secuencia2Completa

  return Secuencia1, Secuencia2

def draw_dotplot(matrix, titulo, fig_name='dotplot.svg'):
  plt.figure(figsize=(5,5))
  plt.imshow(matrix, cmap='Greys',aspect='auto')
  plt.ylabel("Secuencia 1")
  plt.xlabel("Secuencia 2")
  plt.title("Grafica "+ titulo)
  plt.savefig(fig_name)

@nb.njit(parallel=True)
def parallel_image_filter(dotplot):
  filtered_image = np.zeros_like(dotplot, dtype=np.uint8)
  rows, cols = dotplot.shape

  for i in nb.prange(rows):
    for j in range(cols):
      if dotplot[i, j] <= 0:
        filtered_image[i, j] = 255  # Asignar blanco
      elif i == j:
        filtered_image[i, j] = 0  # Asignar negro en diagonales
      else:
        #filtered_image[i, j] = dotplot[i, j]  # Asignar valores originales en grises
        filtered_image[i, j] = 128

  return filtered_image

def show_filtered_image(filter, fig_name='dotplot_fill.svg'):
  plt.imshow(filter, cmap='gray')
  plt.savefig(fig_name)

def dotplot_secuencial(sec1, sec2):
  print("Dotplot Secuencial")
  dotplot = np.empty([len(sec1),len(sec2)], dtype=np.uint64)

  for i in tqdm(range(dotplot.shape[0])):
    for j in range(dotplot.shape[1]):
      if sec1[i] == sec2[j]:
        dotplot[i,j] = 1
      else:
        dotplot[i,j] = 0
  #draw_dotplot(dotplot,"Secuencial", "dotplot_secuencial.png")
  return dotplot

def worker(args):
  i, Secuencia1, Secuencia2 = args
  return [Secuencia1[i] == Secuencia2[j] for j in range(len(Secuencia2))]

def parallel_dotplot(Secuencia1, Secuencia2, threads):
  with mp.Pool(processes=threads) as pool:
    result = pool.map(worker, [(i, Secuencia1, Secuencia2) for i in range(len(Secuencia1))])
  dotplot = np.array(result, dtype=bool)
  return dotplot

def dotplot_parallel_multiprocessing(sec1,sec2, threads):
  print("Dotplot multiprocesamiento")
  dotplot = np.array(parallel_dotplot(sec1, sec2,threads))
  #draw_dotplot(dotplot,"multiprocessing", "dotplot_multip.png")
  #filtered_image = parallel_image_filter(dotplot)
  #show_filtered_image(filtered_image, "dotplot_multip_fill.png")
  return dotplot

def dotplot_parallel_mpi4py(sec1,sec2, threads):
  chunks = np.array_split(range(len(sec1)), threads)

  dotplot = np.empty([len(chunks[rank]),len(sec2)], dtype = np.uint64)

  print("Dotplot MPI4PY")
  for i in tqdm(range(len(chunks[rank]))):
    for j in range(len(sec2)):
      if sec1[chunks[rank][i]] == sec2[j]:
        dotplot[i,j] = np.uint64(1)
      else:
        dotplot[i,j] = np.uint64(0)

  gather_dotplot = comm.gather(dotplot, root=0)

  if rank == 0:
    merged_data = np.vstack(gather_dotplot)
  #draw_dotplot(merged_data, "MPI4PY",fig_name='dotplot_paralell_mpi4py.png')
  #filtered_image = parallel_image_filter(merged_data)
  #show_filtered_image(filtered_image, "dotplot_mpi4_fill.png")

  return merged_data

def run_dotplot(sec1, sec2,  threads):

  start_time = time.time()
  dotplot_seq = dotplot_secuencial(sec1, sec2)
  secuential_time = time.time() - start_time

  start_time = time.time()
  dotplot_mp = dotplot_parallel_multiprocessing(sec1, sec2, threads)
  multiprocessing_time = time.time() - start_time

  start_time = time.time()
  dotplot_mpi = dotplot_parallel_mpi4py(sec1, sec2, threads)
  mpi_time = time.time() - start_time

  if rank == 0:
    print(f"Tiempo Secuencial: {secuential_time} segundos")
    print(f"Tiempo multiprocesamiento: {multiprocessing_time} segundos")
    print(f"Tiempo MPI: {mpi_time} segundos")
    #print(f"Filtrado de imagen: {filtering_time} segundos")

    #Desempeño del código
    total_time = secuential_time + multiprocessing_time + mpi_time
    parallelizable_time = multiprocessing_time + mpi_time
    sequential_time = secuential_time
    #image_generation_time = filtering_time
    idle_time = total_time -(parallelizable_time)

    print(f"Tiempo paralelizable: {parallelizable_time} segundos")
    print(f"Tiempo secuencial: {secuential_time} segundos")
    print(f"tiempo muerto: {idle_time} segundos")
    print(f"Tiempo total:{total_time} segundos")
    #print(f"Tiempo de generación de imagen: {image_generation_time}")

    items.append([secuential_time, multiprocessing_time, mpi_time])
    totals.append(total_time)

    return totals, items, dotplot_seq, dotplot_mp, dotplot_mpi

def plot_performance(times, threads):
  plt.figure(figsize=(12, 4))
  labels = ['SEQ', 'MP', 'MPI']
  plt.xlabel('Cantidad de hilos/procesos')
  plt.ylabel('Tiempo (segundos)')

  for x in range(len(threads)):
      plt.subplot(1, len(threads), x+1)
      plt.bar(labels, times[x],linewidth=0.3)
      plt.xlabel("Cores: "+ str(threads[x]))
      plt.ylabel('Tiempo (segundos)')
      plt.title('Tiempos de ejecución')
      plt.xticks(rotation=45)

      # Obtener la cantidad de hilos para el gráfico actual
      #thread_count = threads[x]

      # Agregar la cantidad de hilos debajo de las etiquetas
      #plt.text(0, -0.6, f'{thread_count} hilos', ha='center')

  plt.tight_layout()  # Ajustar el espaciado entre los gráficos
  plt.savefig('tiempos_ejecucion.png')


def plot_accel_effic(times, threads):
  acel = [times[0]/thread for thread in times]
  efic = [acel[i]/threads[i] for i in range(len(threads))]
  plt.figure()
  plt.plot(threads, acel)
  plt.plot(threads, efic)
  plt.xlabel('Cantidad de hilos/procesos')
  plt.ylabel('Aceleración y eficiencia')
  plt.legend(["Aceleración"," eficiencia"])
  plt.savefig('acelyefic.png')

def plot_scalability(times, threads):
  plt.figure()
  plt.plot(threads, times, marker='o')
  plt.xlabel('Numero de procesadores')
  plt.ylabel('Tiempo de ejecución (segundos)')
  plt.title('Escalabilidad')
  plt.savefig('escalabilidad.png')

def main():

  global comm, size, rank, totals, items
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  parser = argparse.ArgumentParser(description="Ayuda en linea de comandos")
  parser.add_argument("--file1","-f", help="Archivo FASTA con la secuencia de ADN completa, nombre del archivo con extensión fna")
  parser.add_argument("--file2","-g", help="Archivo FASTA con la secuencia de ADN completa, nombre del archivo con extensión fna")
  parser.add_argument("--ps", type=float, help="Porcentaje de la secuencia completa a analizar")
  parser.add_argument("--threads", nargs='+', type=int, default=[1], help="Número de threads para el procesamiento, por defecto se utilizara 1 hilo")
  parser.add_argument('--mpi', action='store_true', help='Usar MPI para el procesamiento paralelo')
  args = parser.parse_args()

  file1 = args.file1
  file2 = args.file2
  porcs = args.ps
  thread_list = args.threads
  times = []
  totals = []
  items = []
  print("Nombre archivo1:",file1)
  print("Nombre archivo2:",file2)
  print("Lista de hilos:", thread_list)

  start_time = time.time()
  sec1, sec2 = cargar_secuencia(file1, file2, porcs)
  load_time = time.time() - start_time

  print(f"Tiempo de carga de archivo: {load_time} segundos")

  for threads in thread_list:
    print("\n")
    print(f"Ejecucion con {threads} hilos")
    make_dotplot = run_dotplot(sec1, sec2, threads)
  #print("resultados dotplot:", make_dotplot)

  start_time = time.time()

  draw_dotplot(make_dotplot[2],"Secuencial", "dotplot_secuencial.png")
  filtered_seq = parallel_image_filter(make_dotplot[2])
  show_filtered_image(filtered_seq, "dotplot_multip_fill.png")

  draw_dotplot(make_dotplot[3],"Multiprocesamiento", "dotplot_multip.png")
  filtered_mp = parallel_image_filter(make_dotplot[3])
  show_filtered_image(filtered_mp, "dotplot_multip_fill.png")

  draw_dotplot(make_dotplot[4], "MPI4PY",fig_name='dotplot_paralell_mpi4py.png')
  filtered_mpi = parallel_image_filter(make_dotplot[4])
  show_filtered_image(filtered_mpi, "dotplot_mpi4_fill.png")

  plot_performance(make_dotplot[1], thread_list)
  plot_scalability(make_dotplot[0], thread_list)
  plot_accel_effic(make_dotplot[0], thread_list)

  generating_images = time.time() - start_time

  print(f"Tiempo de generación de imagenes: {generating_images} segundos")

if __name__ == "__main__":
  main()