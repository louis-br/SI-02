% perceptron para OR com duas entradas

X = [0 0 1 1; 
     0 1 0 1];
% target (y = saidas esperadas)
T = [0 1 1 1];

% Plota X(1) x Xp(2) 
% + sao as saidas = 1
% 0 sao as saidas = 0
plotpv(X,T);

% cria uma rede neural de 1 perceptron
net = perceptron;
net = configure(net,X,T);
% atribui os pesos das duas entradas
net.iw{1,1} = [1 1];
% atribui os peso da entrada biais (limiar)
net.b{1} = -2;

% plota o separador
plotpc(net.iw{1,1},net.b{1});

% simula uma entrada e o perceptron e prediz a saida
x = [1; 1];
y = net(x);
disp("ent:");
disp(x);
disp("saida");
disp(y);
