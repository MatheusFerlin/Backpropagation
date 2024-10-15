string filePath = "IRIS.csv";
//Fazendo a leitura do arquivo e armazenando os dados em um array de IrisData (Record) que foi criado para armazenar os valores de entrada e saída
//Essa lógica está no final do arquivo, onde temos a classe IrisDataReader
IrisData[] irisData = IrisDataReader.ReadData(filePath);
//Esses valores estão dinâmicos, caso exista a necessidade, alterá-los irá mudar a forma de treinamento da rede. 
//Pelo que percebi, não existe a necessidade de 10 000 épocas para esse dataset.
//Deixe esse número alto, mas coloquei uma condição de parada no treinamento, para garantir que não ocorra overfitting
double taxaAprendizado = 0.1;
int numEpocas = 10000;
int totalData = irisData.Length;
int trainDataSize = totalData / 4;

//Ajustando a proporção do conjunto de treinamento em relação ao conjunto de teste
IrisData[] trainData = new IrisData[trainDataSize * 3];
IrisData[] testData = new IrisData[totalData - trainDataSize * 3];
int trainIndex = 0;
int testIndex = 0;

// Dividir os dados entre treino e teste
//Essa lógica garante que teram elementos de todas as classes no conjunto de treinamento e de testes.
for (int i = 0; i < totalData - 1; i++)
{
    if (irisData[i].Saida == 0 && trainIndex < trainDataSize)
    {
        trainData[trainIndex++] = irisData[i];
    }
    else if (irisData[i].Saida == 1 && trainIndex < trainDataSize * 2)
    {
        trainData[trainIndex++] = irisData[i];
    }
    else if (irisData[i].Saida == 2 && trainIndex < trainDataSize * 3)
    {
        trainData[trainIndex++] = irisData[i];
    }
    else
    {
        testData[testIndex++] = irisData[i];
    }
}

// Chamada para instanciar a classe do classificador, o 3 representa a quantidade de classes das nossas saídas, nesse caso está fixo para funcionar com 3
MLPClassifier backpropagation = new(3);

Random random = new();
//Embaralhando os dados de treinamento, para garantir que a rede não aprenda a ordem dos dados
trainData = trainData.OrderBy(x => random.Next()).ToArray();

//Realizando o treinamento da rede
backpropagation.Fit(
    trainData.Select(d => d.Entradas).ToArray(),
    trainData.Select(d => d.Saida).ToArray(),
    taxaAprendizado, numEpocas
);

double[][] trainDataInputs = trainData.Where(data => data != null).Select(data => data.Entradas).ToArray();
int[] trainPredictions = backpropagation.Predict(trainDataInputs);
int[] trainTrueLabels = trainData.Where(data => data != null).Select(data => data.Saida).ToArray();

//Validando a performance da rede com os dados que foram utilizados para o treinamento, para termos uma ideia de como ele se comporta em relação aos dados de teste
Console.WriteLine("-------------------------------------------------------------------------");
Console.WriteLine("Análise dos dados de treinamento:");
Console.WriteLine("-------------------------------------------------------------------------");

double trainAccuracy = MLPClassifier.Acuracia(trainPredictions, trainTrueLabels);
double trainPrecision0 = MLPClassifier.Precisao(trainPredictions, trainTrueLabels, 0);
double trainPrecision1 = MLPClassifier.Precisao(trainPredictions, trainTrueLabels, 1);
double trainPrecision2 = MLPClassifier.Precisao(trainPredictions, trainTrueLabels, 2);
double trainRecall0 = MLPClassifier.Recall(trainPredictions, trainTrueLabels, 0);
double trainRecall1 = MLPClassifier.Recall(trainPredictions, trainTrueLabels, 1);
double trainRecall2 = MLPClassifier.Recall(trainPredictions, trainTrueLabels, 2);
double trainF1Score1 = MLPClassifier.F1Score(trainPredictions, trainTrueLabels, 0);
double trainF1Score2 = MLPClassifier.F1Score(trainPredictions, trainTrueLabels, 1);
double trainF1Score3 = MLPClassifier.F1Score(trainPredictions, trainTrueLabels, 2);

Console.WriteLine($"Accuracy: {trainAccuracy}");
Console.WriteLine($"Precision0: {trainPrecision0}");
Console.WriteLine($"Precision1: {trainPrecision1}");
Console.WriteLine($"Precision2: {trainPrecision2}");
Console.WriteLine($"Recall0: {trainRecall0}");
Console.WriteLine($"Recall1: {trainRecall1}");
Console.WriteLine($"Recall2: {trainRecall2}");
Console.WriteLine($"F1-Score1: {trainF1Score1}");
Console.WriteLine($"F1-Score2: {trainF1Score2}");
Console.WriteLine($"F1-Score3: {trainF1Score3}");

Console.WriteLine("\n");
Console.WriteLine("-------------------------------------------------------------------------");
Console.WriteLine("Análise dos dados de Teste:");
Console.WriteLine("-------------------------------------------------------------------------");

double[][] testDataInputs = testData.Where(data => data != null).Select(data => data.Entradas).ToArray();
int[] predictions = backpropagation.Predict(testDataInputs);
int[] trueLabels = testData.Where(data => data != null).Select(data => data.Saida).ToArray();

// Avaliação de desempenho dos dados de teste, esses dados não foram utilizados no treinamento
//A função Zip está sendo utilizada para verificar se o valor previsto é igual ao valor real, caso seja, incrementa o truePositives
double accuracy = MLPClassifier.Acuracia(predictions, trueLabels);
double precision0 = MLPClassifier.Precisao(predictions, trueLabels, 0);
double precision1 = MLPClassifier.Precisao(predictions, trueLabels, 1);
double precision2 = MLPClassifier.Precisao(predictions, trueLabels, 2);
double recall0 = MLPClassifier.Recall(predictions, trueLabels, 0);
double recall1 = MLPClassifier.Recall(predictions, trueLabels, 1);
double recall2 = MLPClassifier.Recall(predictions, trueLabels, 2);
double f1Score1 = MLPClassifier.F1Score(predictions, trueLabels, 0);
double f1Score2 = MLPClassifier.F1Score(predictions, trueLabels, 1);
double f1Score3 = MLPClassifier.F1Score(predictions, trueLabels, 2);

Console.WriteLine($"Accuracy: {accuracy}");
Console.WriteLine($"Precision0: {precision0}");
Console.WriteLine($"Precision1: {precision1}");
Console.WriteLine($"Precision2: {precision2}");
Console.WriteLine($"Recall0: {recall0}");
Console.WriteLine($"Recall1: {recall1}");
Console.WriteLine($"Recall2: {recall2}");
Console.WriteLine($"F1-Score1: {f1Score1}");
Console.WriteLine($"F1-Score2: {f1Score2}");
Console.WriteLine($"F1-Score3: {f1Score3}");

Console.ReadLine();

public class MLPClassifier
{
    /*Professor, deixei o programa fixo, aqui temos 3 camadas escondidas com 17 neurônios cada (Inicialmente eu fiz ao contrário, mas você me auxiliou pelo Whatsapp e percebi o erro)
     * Caso seja necessário alterar a quantidade de camadas ou neurônios, vai ser necessário mexer na lógica do programa
    */
    private readonly int inputSize;
    private readonly int outputSize;
    private readonly double[][][] pesos; // Pesos das conexões entre as camadas
    private readonly double[][] bias;
    private readonly int[] camadas = { 17, 17, 17 }; // Aqui temos a declaração das 3 camadas escondidas da rede, com 17 neurônios cada
    private readonly int numClasses; // Número de classes do problema

    public MLPClassifier(int numClasses)
    {
        this.inputSize = 4; // 4 características do dataset Iris
        this.outputSize = numClasses; // 3 classes (0, 1, 2) Tratei o dataset para trocar o nome das classes para números inteiros, para facilitar no código
        this.numClasses = numClasses;

        pesos = new double[4][][]; // 3 camadas escondidas + camada de saída (Inicialmente eu também havia esquecido de deixar uma camada de saída, o que também deu problema)
        bias = new double[4][]; //Erro para cada camada
        //Inicializando os pesos de forma aleatória
        InicializarPesos();
    }
    private void InicializarPesos()
    {
        Random rand = new();

        // Inicialização dos pesos e bias
        for (int c = 0; c < 4; c++)
        {
            int entradas = c == 0 ? inputSize : camadas[c - 1]; //Se for a primeira camada, a quantidade de entradas é o tamanho da entrada (4), senão é 17, o número de neurônios.
            int saidas = c == 3 ? outputSize : camadas[c]; //Se for a última camada, a quantidade de saídas é o número de classes (3), senão é 17, o número de neurônios.

            pesos[c] = new double[saidas][];
            bias[c] = new double[saidas];

            for (int i = 0; i < saidas; i++)
            {
                pesos[c][i] = new double[entradas];
                for (int j = 0; j < entradas; j++)
                    pesos[c][i][j] = rand.NextDouble() * 2 - 1; // Pesos aleatórios em [-1, 1], a função rand.NextDouble() retorna um número de 0 a 1
                bias[c][i] = rand.NextDouble() * 2 - 1; // Bias aleatório em [-1, 1]
            }
        }
    }
    //Foi escolhida a função de ativação sigmoide, com base no Slide na aula. A formulá foi transcrita de lá
    private static double Sigmoide(double x) => 1.0 / (1.0 + Math.Exp(-x));

    //A derivada da função sigmóide também foi retirada dos slides
    private static double SigmoideDerivada(double y) => y * (1 - y);

    public void Fit(double[][] entradas, int[] saidasEsperadas, double taxaAprendizado, int numEpocas)
    {
        for (int epoca = 0; epoca < numEpocas; epoca++)
        {
            double erroTotal = 0.0;

            for (int i = 0; i < entradas.Length; i++)
            {
                //Inicialmente calcula o resultado com base nos parâmetros atuais; (Conforme algoritmo)
                double[] saida = FeedForward(entradas[i]);

                // Cálculo do erro total
                for (int j = 0; j < saida.Length; j++)
                {
                    int expectedOutput = (saidasEsperadas[i] == j) ? 1 : 0; // One-hot encoding (Vi esse algoritmo na internet, colocando 1 na posição correpondente a classe desejada 
                    erroTotal += Math.Pow(expectedOutput - saida[j], 2); // Calculando o erro total pelos erros quadráticos, também retirado dos slides da aula 2
                }
                //Fazendo a retroalimentação após calcular o erro para cada entrada, conforme algoritmo visto na aula
                Backpropagate(entradas[i], saidasEsperadas[i], taxaAprendizado);
            }

            Console.WriteLine($"Época {epoca + 1}, Erro Total: {erroTotal}");

            // Interrompe o treinamento se o erro total for menor que 4 (Valor que eu escolhi para garantir que o treinamento não ocorra overfitting)
            //Caso queira, pode alterar ou remover esse valor.
            if (erroTotal < 4)
            {
                Console.WriteLine($"Treinamento interrompido na época {epoca + 1} devido ao erro total menor que 4.");
                Console.WriteLine("\n");
                break;
            }
        }
    }

    ///<summary>
    /// Realiza o cálculo da ativação da rede neural para uma determinada entrada.
    /// </summary>
    /// <param name="entrada">As entradas da rede neural.</param>
    /// <returns>Um array com os valores de ativação para cada neurônio na camada de saída.</returns>
    private double[] FeedForward(double[] entrada)
    {
        double[] ativacao = entrada;

        for (int c = 0; c < 4; c++)
        {
            // Faz o cálculo da ativação com base nos pesos e erros atuais.
            ativacao = CalcularAtivacao(ativacao, pesos[c], bias[c]);
        }
        return ativacao;
    }

    /// <summary>
    /// Realiza a retropropagação do erro para atualizar os pesos da rede neural.
    /// </summary>
    /// <param name="entrada">As entradas da rede neural.</param>
    /// <param name="esperado">A saída esperada da rede neural.</param>
    /// <param name="taxaAprendizado">A taxa de aprendizado da rede neural.</param>
    private void Backpropagate(double[] entrada, int esperado, double taxaAprendizado)
    {
        double[][] ativacoes = new double[4][];
        double[][] erros = new double[4][];

        ativacoes[0] = CalcularAtivacao(entrada, pesos[0], bias[0]);

        for (int c = 1; c < 4; c++)
        {
            ativacoes[c] = CalcularAtivacao(ativacoes[c - 1], pesos[c], bias[c]);
        }

        // Erros para a camada de saída
        //Validando os erros apenas para a camada 3, que é a camada de saída, novamente, deixei os valores fixos para funcionar com o dataset Iris
        erros[3] = new double[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            //Nesse caso o esperado é a saída com base nas entradas do dataset (Temos esse dado no dataset Iris)
            erros[3][i] = (esperado == i ? 1.0 : 0.0) - ativacoes[3][i]; // Erro da saída
        }

        // Erros para as camadas escondidas
        //Aqui calculamos os erros para as camadas escondidas, que são as camadas 2, 1 e 0, calculamos de trás para frente conforme o algoritmo.
        for (int c = 2; c >= 0; c--)
        {
            erros[c] = new double[camadas[c]];
            for (int i = 0; i < camadas[c]; i++)
            {
                double erro = 0.0;
                for (int j = 0; j < erros[c + 1].Length; j++)
                {
                    erro += erros[c + 1][j] * pesos[c + 1][j][i];
                }
                erros[c][i] = erro * SigmoideDerivada(ativacoes[c][i]);
            }
        }
        //Ao final, atualizamos os pesos com base nos novos erros (bias)
        AtualizarPesos(entrada, ativacoes, erros, taxaAprendizado);
    }

    private void AtualizarPesos(double[] entrada, double[][] ativacoes, double[][] erros, double taxaAprendizado)
    {
        for (int c = 0; c < 4; c++)
        {
            double[] input = c == 0 ? entrada : ativacoes[c - 1];

            for (int i = 0; i < pesos[c].Length; i++)
            {
                for (int j = 0; j < pesos[c][i].Length; j++)
                {
                    //Atualizando os pesos para cada camada, C representa a camada                   
                    pesos[c][i][j] += taxaAprendizado * erros[c][i] * input[j];
                }
                //Atualizando o erro para cada camada
                bias[c][i] += taxaAprendizado * erros[c][i];
            }
        }
    }
    /// <summary>
    /// Calcula a ativação de cada neurônio em uma camada.
    /// </summary>
    /// <param name="entrada">As entradas para a camada.</param>
    /// <param name="pesos">Os pesos das conexões entre os neurônios.</param>
    /// <param name="bias">Os valores de bias para cada neurônio.</param>
    /// <returns>Um array com os valores de ativação para cada neurônio na camada.</returns>
    private static double[] CalcularAtivacao(double[] entrada, double[][] pesos, double[] bias)
    {
        // Aqui podemos ver a aplicação do Perceptron, onde a ativação é calculada com base na soma dos pesos e bias
        double[] ativacao = new double[pesos.Length];
        for (int i = 0; i < pesos.Length; i++)
        {
            double soma = bias[i];
            for (int j = 0; j < entrada.Length; j++)
                soma += pesos[i][j] * entrada[j];
            ativacao[i] = Sigmoide(soma);
        }
        return ativacao;
    }

    /// <summary>
    /// M[etodo para prever a saída para um conjunto de entradas.
    /// </summary>
    /// <param name="entradas"></param>
    /// <returns>As saídas previstas para cada elemento</returns>
    public int[] Predict(double[][] entradas)
    {
        int[] saidasPrevistas = new int[entradas.Length];

        for (int i = 0; i < entradas.Length; i++)
        {
            double[] ativacao = FeedForward(entradas[i]);
            // Obtém a classe prevista com base na maior ativação
            saidasPrevistas[i] = Array.IndexOf(ativacao, ativacao.Max());
        }

        return saidasPrevistas;
    }
    /// Trechos de código para calcular as métricas base para avaliação de classificadores
    public static double Acuracia(int[] preditos, int[] reais)
    {
        int acertos = preditos.Zip(reais, (predito, real) => predito == real ? 1 : 0).Sum();
        return (double)acertos / reais.Length;
    }

    public static double Precisao(int[] preditos, int[] reais, int classe)
    {
        int verdadeirosPositivos = preditos.Zip(reais, (predito, real) => (predito == classe && real == classe) ? 1 : 0).Sum();
        int falsosPositivos = preditos.Zip(reais, (predito, real) => (predito == classe && real != classe) ? 1 : 0).Sum();
        return verdadeirosPositivos + falsosPositivos > 0 ? (double)verdadeirosPositivos / (verdadeirosPositivos + falsosPositivos) : 0;
    }

    public static double Recall(int[] preditos, int[] reais, int classe)
    {
        int verdadeirosPositivos = preditos.Zip(reais, (predito, real) => (predito == classe && real == classe) ? 1 : 0).Sum();
        int falsosNegativos = preditos.Zip(reais, (predito, real) => (predito != classe && real == classe) ? 1 : 0).Sum();
        return verdadeirosPositivos + falsosNegativos > 0 ? (double)verdadeirosPositivos / (verdadeirosPositivos + falsosNegativos) : 0;
    }

    public static double F1Score(int[] preditos, int[] reais, int classe)
    {
        double prec = Precisao(preditos, reais, classe);
        double rec = Recall(preditos, reais, classe);
        return prec + rec > 0 ? 2 * (prec * rec) / (prec + rec) : 0;
    }
}

//Record utilizado para a leitura do Dataset Iris
public record IrisData(double[] Entradas, int Saida);

public class IrisDataReader
{
    //Implementação apenas para realizar a leitura do arquivo, percorre todas as linhas e separa os valores de entrada e saída
    public static IrisData[] ReadData(string filePath)
    {
        IrisData[] data;
        using (var reader = new StreamReader(filePath))
        {
            var text = reader.ReadToEnd().Replace("\r", "");
            var lines = text.Split('\n');
            data = new IrisData[lines.Length - 1];

            for (int i = 1; i < lines.Length - 1; i++)
            {
                var values = lines[i].Split(',');
                double[] inputs = new double[values.Length - 1];
                for (int j = 0; j < values.Length - 1; j++)
                {
                    inputs[j] = double.Parse(values[j].Replace(".", ","));
                }
                int output = int.Parse(values[values.Length - 1]);
                data[i - 1] = new IrisData(inputs, output);
            }
        }
        return data;
    }
}


