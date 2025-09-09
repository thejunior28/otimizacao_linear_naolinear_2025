clear;
clc;

% Matrizes e vetores da função objetivo
[A, arrows, cols, entries, rep, field, symm] = mmread('bcsstk08.mtx'); % Matriz de restrições

u = randn(size(A, 2), 1);
b = A * u;
c = abs(rand(size(A, 2), 1));

A=ichol(A);

% Inicialização para o problema primal
AtA_inv = (A * A') \ b; % Resolver At(AAt)^(-1)b
x_tilde = A' * AtA_inv; % Calcular x_tilde (aproximação para x)

% Ajuste de x_tilde para garantir positividade
epsilon2 = 100;
epsilon1 = max([-min(x_tilde), epsilon2, norm(b, 1) / (epsilon2 * norm(A, 1))]);
x0 = max(x_tilde, epsilon1);

% Inicialização para o problema dual
y0 = zeros(size(A, 2), 1); % Inicializar y0 como zero

% Calcular z0 com base em c
epsilon3 = 1 + norm(c, 1);
z0 = zeros(size(c));
for i = 1:length(c)
    if c(i) >= 0
        z0(i) = c(i) + epsilon3;
    elseif c(i) <= -epsilon3
        z0(i) = -c(i);
    else
        z0(i) = epsilon3;
    end
end

tol=1e-4; % Tolerância para convergência

fprintf("Escolha um método de resolução.\n");
fprintf("Método Primal-Dual Afim-Escala (1), Método Primal-Dual Clássico (2), Método Preditor-Corretor (3),\n");
met_res = input("Método Barreira Logarítmica (4), Método Barreira Logarítmica Preditor-Corretor (5): ");

met_res = round(met_res);

if met_res > 5 || met_res < 1
    fprintf("\nRecomendo escolher um dos métodos propostos.\n");
    return
end

if met_res == 1

    fprintf("\nMétodo Primal-Dual Afim-Escala.\n");

    tau = 1e-4; % Tamanho de passo

    fprintf("\nTau: %d\n",tau);
    
    %A=ichol(A);
    x = x0;
    y = y0;
    z = z0;
    iter = 1;
    fac_primal = norm(b-A*x)/(norm(b)+1);
    fac_dual = norm(c-A'*y-z)/(norm(c)+1);
    gap = abs(x'*z/(1+c'*x+b'*y));

    func_values = zeros(0, 1); % Vetor coluna vazio
    grad_fac_primal = zeros(0, 1);
    grad_fac_dual = zeros(0, 1);
    grad_gap = zeros(0, 1);

    tstart = tic;

    while (gap > tol ||fac_primal > tol || fac_dual > tol)

        func_values = [func_values; c' * x]; % Adicionar ao vetor coluna
        grad_fac_primal = [grad_fac_primal; norm(b - A * x) / (norm(b) + 1)];
        grad_fac_dual = [grad_fac_dual; norm(c - A' * y - z) / (norm(c) + 1)];
        grad_gap = [grad_gap; abs(c' * x - b' * y) / (1 + c' * x + b' * y)];

        % Calculate scaling matrices
        X = diag;
        Z = diag(z);
        D=diag(x./z);

        % Calculate residuals
        rp = b - A * x;
        rd = c - A' * y - z;
        ra = -diag * diag(z) * ones(cols, 1);

        % Calculate search directions
        dx = D \ (A' * ((A * (D * A')) \ (rp + A * (D \ rd) - A * (D \ (diag(1 ./ x) * ra)))));
        dz = D \ (ra - diag(z) * dx);
        dy = (A * (D * A')) \ (rp + A * (D \ rd) - A * (D \ (diag(1 ./ x) *ra)));

        % Calculate step lengths
        rho_p = min(-x(dx < 0) ./ dx(dx < 0));
        rho_d = min(-z(dz < 0) ./ dz(dz < 0));

        alpha_p = min(1, tau * rho_p); 
        alpha_d = min(1, tau * rho_d);

        % Update solutions
        x(x>0) = x + alpha_p * dx; 
        y = y + alpha_d * dy;
        z(z>0) = z + alpha_d * dz;

        fac_primal = norm(b-A*x)/(norm(b)+1);
        fac_dual = norm(c-A'*y-z)/(norm(c)+1);
        gap = abs(c' * x - b' * y) / (1 + abs(c' * x) + abs(b' * y));

        iter = iter + 1;

        if toc(tstart) >= 14400 || isempty(alpha_p) || isempty(alpha_d)
            fo=c'*x;
            telapsed = toc(tstart);
            fprintf("\nNão convergiu!\n")
            fprintf("Número de iterações: %2.f \n", iter);
            fprintf("Função objetivo = %d \n", fo);
            fprintf("Tempo computacional: %d segundos \n", telapsed);
            fprintf("Factibilidade primal: %d \n", fac_primal);
            fprintf("Factibilidade dual: %d \n", fac_dual);
            fprintf("Gap relativo: %d \n", gap);

            % Gráfico de Convergência
            figure;
            subplot(2, 2, 1);
            plot(func_values, '-o');
            title('Função Objetivo');
            xlabel('Iterações');
            ylabel('Valor da Função Objetivo');
        
            subplot(2, 2, 2);
            plot(grad_fac_primal, '-o');
            title('Convergência da Factibilidade primal');
            xlabel('Iterações');
            ylabel('Factibilidade primal');
        
            subplot(2, 2, 3);
            plot(grad_fac_dual, '-o');
            title('Convergência da Factibilidade dual');
            xlabel('Iterações');
            ylabel('Factibilidade dual');
        
            subplot(2, 2, 4);
            plot(grad_gap, '-o');
            title('Convergência do Gap Relativo');
            xlabel('Iterações');
            ylabel('Gap Relativo');
        
            return
        end
    end

    fo= c'*x;

    telapsed = toc(tstart);

    fprintf(" \nSolução: \n");
    fprintf("Número de iterações: %2.f \n", iter);
    fprintf("Função objetivo = %d \n", fo);
    fprintf("Tempo computacional: %d segundos \n", telapsed);
    fprintf("Factibilidade primal: %d \n", fac_primal);
    fprintf("Factibilidade dual: %d \n", fac_dual);
    fprintf("Gap relativo: %d \n", gap);

    % Gráfico de Convergência
    figure;
    subplot(2, 1, 1);
    plot(func_values, '-o');
    title('Função Objetivo');
    xlabel('Iterações');
    ylabel('Valor da Função Objetivo');

    subplot(2, 1, 2);
    plot(grad_fac_primal, '-o');
    title('Convergência da Factibilidade primal');
    xlabel('Iterações');
    ylabel('Factibilidade primal');

    subplot(2, 1, 2);
    plot(grad_fac_dual, '-o');
    title('Convergência da Factibilidade dual');
    xlabel('Iterações');
    ylabel('Factibilidade dual');

    subplot(2, 1, 2);
    plot(grad_gap, '-o');
    title('Convergência do Gap Relativo');
    xlabel('Iterações');
    ylabel('Gap Relativo');

    return

elseif met_res == 2

   fprintf("\nMétodo Primal-Dual Clássico.\n");

tstart = tic;

tau = 0.01; % Valor sugerido

fprintf("\nTau: %d\n",tau);

n = length(x0); % Tamanho do problema
sigma = 1 / sqrt(n); % Valor sugerido
x = x0;
y = y0;
z = z0; % Inicialização
k = 0; % Contador de iterações
gamma_k = 0.9;

func_values = zeros(0, 1); % Vetor coluna vazio
grad_gamma = zeros(0, 1);

while gamma_k >= tol

    % Escolha do valor de mu
    if gamma_k >= 1
        mu_k = sigma * gamma_k / n;
    else
        mu_k = sigma * (gamma_k^2) / n;
    end
    
    func_values = [func_values; c' * x]; % Adicionar ao vetor coluna
    grad_gamma = [grad_gamma; gamma_k*((1+0.05)^(-1))];

    
    % Resíduos
    rp = b - A * x;
    rd = c - A' * y - z;
    rc = mu_k * ones(n, 1) - x .* z;

    % Matrizes diagonais
    Dk = diag(z ./ x);

    % Direções de Newton
    dy = (A * (diag(1 ./ diag(Dk))) * A') \ (rp + A * (diag(1 ./ diag(Dk))) * rd - A * (diag(1 ./ z)) * rc);
    dx = (diag(1 ./ diag(Dk))) * (A' * dy - rd + (diag(1 ./ x)) * rc);
    dz = (diag(1 ./ x)) * (rc - z .* dx);

    % Cálculo dos passos primais e duais
    rho_p = min([-x(dx < 0) ./ dx(dx < 0); 1]);
    rho_d = min([-z(dz < 0) ./ dz(dz < 0); 1]);
    alpha_p = min(1, tau * rho_p);
    alpha_d = min(1, tau * rho_d);

    % Atualização das variáveis
    x = x + alpha_p * dx;
    y = y + alpha_d * dy;
    z = z + alpha_d * dz;

    % Atualiza gamma_k
    gamma_k = gamma_k*((1+0.05)^(-1));

    % Incrementa o contador de iterações
    k = k + 1;

    if toc(tstart) >= 14400 || isempty(gamma_k)
        fo = c' * x;
        telapsed = toc(tstart);
        fprintf("\nNão convergiu!\n");
        fprintf(" \nSolução: \n");
        fprintf("Número de iterações: %2.f \n", k);
        fprintf("Função objetivo = %d \n", fo);
        fprintf("Tempo computacional: %d segundos \n", telapsed);
        fprintf("Mi = %d \n",mu_k);
        fprintf("Gama = %d \n",gamma_k);
        
        % Gráfico de Convergência
        figure;
        subplot(2, 1, 1);
        plot(func_values, '-o');
        title('Função Objetivo');
        xlabel('Iterações');
        ylabel('Valor da Função Objetivo');

        subplot(2, 1, 2);
        plot(grad_gamma, '-o');
        title('Convergência de Gama');
        xlabel('Iterações');
        ylabel('Gama');
        
        return;
    end
end

fo = c' * x;
telapsed = toc(tstart);

fprintf(" \nSolução: \n");
fprintf("Número de iterações: %2.f \n", k);
fprintf("Função objetivo = %d \n", fo);
fprintf("Tempo computacional: %d segundos \n", telapsed);
fprintf("Mi = %d \n", mu_k);
fprintf("Gama = %d \n",gamma_k);

% Gráfico de Convergência
figure;
subplot(2, 1, 1);
plot(func_values, '-o');
title('Função Objetivo');
xlabel('Iterações');
ylabel('Valor da Função Objetivo');

subplot(2, 1, 2);
plot(grad_gamma, '-o');
title('Convergência de Gama');
xlabel('Iterações');
ylabel('Gama');

return

elseif met_res == 3

    fprintf("\nMétodo Preditor-Corretor.\n");

    % Dados iniciais
    x = x0;
    y = y0;
    z = z0;
    iter = 0;
    n = length(x0);

    tau = 0.5; % Valor sugerido

    fprintf("\nTau: %d\n",tau);


    % Inicialização de parâmetros
    gamma_k = (x' * z) / n;

    func_values = zeros(0, 1); % Vetor coluna vazio
    grad_gamma = zeros(0, 1);

    tstart = tic;

    while gamma_k > tol

        func_values = [func_values; c' * x]; % Adicionar ao vetor coluna
        grad_gamma = [grad_gamma; (x' * z) / n];

        % Resíduos

        rp = b - A * x;
        rd = c - A' * y - z;
        ra = -diag * diag(z) * ones(n, 1);

 % Direções preliminares
        Dk = diag(x ./ z);
        d_tilde_y = (A * (Dk \ A')) \ (rp + A * (Dk \ rd) - A * (diag(1 ./ x) * ra));
        d_tilde_x = Dk \ (A' * d_tilde_y - rd + diag(1 ./ x) * ra);
        d_tilde_z = diag(1 ./ x) * (ra - diag(z) * d_tilde_x);

        % Tamanhos de passo preliminares
        rho_tilde_p = min(-x(d_tilde_x < 0) ./ d_tilde_x(d_tilde_x < 0));
        rho_tilde_d = min(-z(d_tilde_z < 0) ./ d_tilde_z(d_tilde_z < 0));

        if isempty(rho_tilde_p)
            alpha_tilde_p=1;
        else
            alpha_tilde_p = min(1, tau * rho_tilde_p);
        end

        if isempty(rho_tilde_d)
            alpha_tilde_d=1;
        else
            alpha_tilde_d = min(1, tau * rho_tilde_d);
        end
        

        % Parâmetro centralizador
        gamma_tilde_k = (x + alpha_tilde_p * d_tilde_x)' * (z + alpha_tilde_d * d_tilde_z);
        
        if gamma_k > 1
            sigma_k = (gamma_tilde_k / gamma_k)^3;
        else
            sigma_k = gamma_k / sqrt(n);
        end

        mu_k = sigma_k * (gamma_k / n);
        
        x_tilde = x + alpha_tilde_p * d_tilde_x;
        z_tilde = x + alpha_tilde_d * d_tilde_z;

        % Resíduo centralizado
        rs = ra + mu_k * ones(n, 1) - diag(x_tilde) * diag(z_tilde) * ones(n,1);

        % Direções finais
        dy = (A * (Dk \ A')) \ (rp + A * (Dk \ rd) - A * (diag(1 ./ x) * rs));
        dx = Dk \ (A' * dy - rd + diag(1 ./ x) * rs);
        dz = diag(1 ./ x) * (rs - diag(z) * dx);

        % Tamanhos de passo finais
        rho_p = min(-x(dx < 0) ./ dx(dx < 0));
        rho_d = min(-z(dz < 0) ./ dz(dz < 0));
        
        if isempty(rho_p)
            alpha_p=1;
        else
            alpha_p = min(1, tau * rho_p);
        end

        if isempty(rho_d)
            alpha_d=1;
        else
            alpha_d = min(1, tau * rho_d);
        end

        % Atualização das variáveis
        x = x + alpha_p * dx;
        y = y + alpha_d * dy;
        z = z + alpha_d * dz;

    % Atualização de parâmetros
    gamma_k = (x' * z) / n;

    if toc(tstart) >= 14400 || gamma_k<0
        fo = c' * x;
        telapsed = toc(tstart);
        fprintf("\nNão convergiu!\n");
        fprintf("Número de iterações: %2.f \n", iter);
        fprintf("Função objetivo = %d \n", fo);
        fprintf("Tempo computacional: %d segundos \n", telapsed);
        fprintf("Gama = %d \n",gamma_k);
        
        % Gráfico de Convergência
        figure;
        subplot(2, 1, 1);
        plot(func_values, '-o');
        title('Função Objetivo');
        xlabel('Iterações');
        ylabel('Valor da Função Objetivo');

        subplot(2, 1, 2);
        plot(grad_gamma, '-o');
        title('Convergência de Gama');
        xlabel('Iterações');
        ylabel('Gama');
        
        return;
    end
    iter=iter+1;
 end

fo = c' * x;
telapsed = toc(tstart);

fprintf(" \nSolução: \n");
fprintf("Número de iterações: %2.f \n", iter);
fprintf("Função objetivo = %d \n", fo);
fprintf("Tempo computacional: %d segundos \n", telapsed);
fprintf("Gama = %d \n",gamma_k);

% Gráfico de Convergência
figure;
subplot(2, 1, 1);
plot(func_values, '-o');
title('Função Objetivo');
xlabel('Iterações');
ylabel('Valor da Função Objetivo');

subplot(2, 1, 2);
plot(grad_gamma, '-o');
title('Convergência de Gama');
xlabel('Iterações');
ylabel('Gama');
return

elseif met_res == 4

    fprintf("\nMétodo Barreira Logarítmica.\n");
    
    tau = 0.5; % Valor sugerido

    fprintf("\nTau: %d\n",tau);

    x = x0;
    s = -A*x+b;
    lambda = rand*ones(cols,1);
    mu = mean(s .* pi); % Inicializa o parâmetro de barreira
    k = 0;
    beta=1.5;
    p_i = mu*diag(1./s)*ones(cols,1);
    max_iter=1e6;

    % Definição de f, g, h e suas derivadas
    f = 0.5 * x' * (A' * A) * x - b' * x; % Função objetivo quadrática
    gradient_f = (A' * A) * x - b; % Gradiente de f
    hessian_f = A' * A; % Hessiana de f
    
    g = x'*A*x-b; % Restrições de desigualdade (g >= 0)
    gradient_g = A*x; % Gradiente de g
    hessian_g = zeros(length(x)); % Hessiana de g (zero pois g é linear)
    
    h = sum(x) - c; % Restrição de igualdade h = sum(x) - c
    gradient_h = ones(size(x)); % Gradiente de h (vetor de 1s)
    hessian_h = zeros(size(x)); % Hessiana de h (matriz zero)

    func_values = zeros(0, 1); % Vetor coluna vazio
    grad_norms = zeros(0, 1); % Vetor coluna vazio

    del_L = norm([gradient_f + gradient_g' * p_i + gradient_h' * lambda;g + s;h;diag(s) * p_i - mu * ones(length(s), 1)]);

    tstart = tic;

while del_L > tol
    % Gradiente da Lagrangiana
    S = diag(s);
    e = ones(length(s), 1);
    grad_L = [gradient_f + p_i'*gradient_g + lambda'*gradient_h; g + s; h; S * p_i - mu * e];
    
    func_values = [func_values; c' * x]; % Adicionar ao vetor coluna
    grad_norms(end + 1) = norm(grad_L);

    % Matriz Hessiana e sistema linear
    H = [hessian_f + p_i' * gradient_g + lambda'*gradient_h, diag(gradient_g), diag(gradient_h), zeros(cols,cols);
     diag(gradient_g), zeros(cols,cols), zeros(cols,cols), eye(cols,cols);
     diag(gradient_h), zeros(cols,cols), zeros(cols,cols), zeros(cols,cols);
     zeros(cols,cols), S, zeros(cols,cols), diag(p_i)];
 
    rhs = -[gradient_f + p_i' * gradient_g + lambda' * gradient_h; g + s; h; S * p_i - mu * e];

    delta = H \ rhs;

    % Divisão do vetor delta
    dx = delta(1:length(x));
    dp_i = delta(length(x)+1:length(x)+length(p_i));
    dlambda = delta(length(x)+length(p_i)+1:length(x)+length(p_i)+length(lambda));
    ds = delta(length(x)+length(p_i)+length(lambda)+1:end);

    % Cálculo de \alpha
    rho_p = min([-s(ds < 0) ./ ds(ds < 0); 1]);
    rho_d = min([-p_i(dp_i < 0) ./ dp_i(dp_i < 0); 1]);
    alpha = min([1, tau * rho_p, tau * rho_d]);

    % Atualização das variáveis
    x = x + alpha * dx;
    s = s + alpha * ds;
    p_i = p_i + alpha * dp_i;
    lambda = lambda + alpha * dlambda;

    % Atualização do parâmetro de barreira
    mu = mu / beta;

    % Definição de f, g, h e suas derivadas
    f = 0.5 * x' * (A' * A) * x - b' * x; % Função objetivo quadrática
    gradient_f = (A' * A) * x - b; % Gradiente de f
    hessian_f = A' * A; % Hessiana de f
    
    g = x'*A*x-b; % Restrições de desigualdade (g >= 0)
    gradient_g = A*x; % Gradiente de g
    hessian_g = zeros(length(x)); % Hessiana de g (zero pois g é linear)
    
    h = sum(x) - c; % Restrição de igualdade h = sum(x) - c
    gradient_h = ones(size(x)); % Gradiente de h (vetor de 1s)
    hessian_h = zeros(size(x)); % Hessiana de h (matriz zero)


    % Incrementa o contador
    k = k + 1;
    fo = c'*x;
    del_L = norm([gradient_f + p_i' * gradient_g + lambda' * gradient_h; g + s; h; diag(s) * p_i - mu * ones(length(s), 1)]);

    if k >= max_iter ||  toc(tstart) >= 14400 || isempty(del_L)
        fprintf("\nNão convergiu!\n")
        fprintf("Número de iterações: %2.f \n", k);
        fprintf("Função objetivo = %d \n", fo);
        fprintf("Delta L = %d \n", del_L);
        fprintf("Tempo computacional: %d segundos \n", telapsed);

        % Gráfico de Convergência
        figure;
        subplot(2, 1, 1);
        plot(func_values, '-o');
        title('Função Objetivo');
        xlabel('Iterações');
        ylabel('Valor da Função Objetivo');
        
        subplot(2, 1, 2);
        plot(grad_norms, '-o');
        xlabel('Iteração');
        ylabel('Norma do Gradiente da Lagrangiana');
        title('Convergência do Algoritmo');
        grid on;
    end
end

    telapsed = toc(tstart);
    
    fprintf("\nSolução:\n")
    fprintf("Número de iterações: %2.f \n", k);
    fprintf("Função objetivo = %d \n", fo);
    fprintf("Delta L = %d \n", del_L);
    fprintf("Tempo computacional: %d segundos \n", telapsed);

    % Gráfico de Convergência
    figure;
    subplot(2, 1, 1);
    plot(func_values, '-o');
    title('Função Objetivo');
    xlabel('Iterações');
    ylabel('Valor da Função Objetivo');
    
    subplot(2, 1, 2);
    plot(grad_norms, '-o');
    xlabel('Iteração');
    ylabel('Norma do Gradiente da Lagrangiana');
    title('Convergência do Algoritmo');
    grid on;

    return

elseif met_res == 5

    fprintf("\nMétodo Barreira Logarítmica Preditor-Corretor.\n");

    tstart = tic;

    % Parâmetros iniciais

    tau = 0.5; % Valor sugerido

    fprintf("\nTau: %d\n",tau);

    x = x0;
    s = -A*x+b;
    lambda = rand*ones(cols,1);
    mu = mean(s .* pi); % Inicializa o parâmetro de barreira
    beta=1.5;
    p_i = mu*diag(1./s)*ones(cols,1);
    max_iter=1e6;

% Definição das funções f, g, h, e suas derivadas
f =  0.5 * x' * (A' * A) * x - b' * x; % Função objetivo quadrática
grad_f = (A' * A) * x - b; % Gradiente de f
hessian_f = A' * A; % Hessiana de f

g = x' * A * x - b; % Restrições de desigualdade (g >= 0)
grad_g = A * x; % Gradiente de g
hessian_g = zeros(length(x)); % Hessiana de g

h = sum(x) - c; % Restrição de igualdade h = sum(x) - c
grad_h = ones(size(x)); % Gradiente de h
hessian_h = zeros(size(x)); % Hessiana de h


del_L= norm([grad_f + p_i' * grad_g + lambda' * grad_h;g + s;h]);

% Iniciar o loop iterativo
k = 0;

func_values = zeros(0, 1); % Vetor coluna vazio
    grad_norms = zeros(0, 1); % Vetor coluna vazio  

    tstart = tic;

while del_L > tol
    % Gradiente da Lagrangiana
    S = diag(s);
    e = ones(length(s), 1);
   
    func_values = [func_values; c' * x]; % Adicionar ao vetor coluna
    grad_norms(end + 1) = norm(del_L);
    % Gradientes da Lagrangiana
    del_L= norm([grad_f + pi' * grad_g + lambda' * grad_h;g + s;h]);
    
    % Matriz W
    W = [hessian_f + pi' * hessian_g + lambda' * hessian_h, diag(grad_g), diag(grad_h), zeros(cols, cols);
         diag(grad_g), zeros(cols, cols), zeros(cols, cols), eye(cols, cols);
         diag(grad_h), zeros(cols, cols), zeros(cols, cols), zeros(cols, cols);
         zeros(cols, cols), diag(s), zeros(cols, cols), diag(p_i)];
    
    % Vetor do lado direito
    rhs = -[grad_f + p_i' * grad_g + lambda' * grad_h;
            g + s;
            h;
            -mu * ones(length(s), 1) + diag(s) * p_i];
    
    % Resolvendo o sistema linear W * [dx; d_pi; d_lambda; d_s] = rhs
    delta = W \ rhs;
    
    % Divisão do vetor delta
    dx = delta(1:length(x));
    dp_i = delta(length(x)+1:length(x)+length(p_i));
    dlambda = delta(length(x)+length(p_i)+1:length(x)+length(p_i)+length(lambda));
    ds = delta(length(x)+length(p_i)+length(lambda)+1:end);
    
    % Cálculo de \alpha
    rho_p = min([-s(ds < 0) ./ ds(ds < 0); 1]);
    rho_d = min([-p_i(dp_i < 0) ./ dp_i(dp_i < 0); 1]);
    alpha = min([1, tau * rho_p, tau * rho_d]);
    
    % Atualização das variáveis
    x = x + alpha * dx;
    s = s + alpha * ds;
    pi = pi + alpha * dp_i;
    lambda = lambda + alpha * dlambda;
    
    % Atualização do parâmetro de barreira
    mu = mu / beta;
    
    % Incrementa o contador
    k = k + 1;
    
    % Cálculo do valor da função objetivo e do critério de convergência
    % Definição das funções f, g, h, e suas derivadas
f =  0.5 * x' * (A' * A) * x - b' * x; % Função objetivo quadrática
grad_f = (A' * A) * x - b; % Gradiente de f
hessian_f = A' * A; % Hessiana de f

g = x' * A * x - b; % Restrições de desigualdade (g >= 0)
grad_g = A * x; % Gradiente de g
hessian_g = zeros(length(x)); % Hessiana de g

h = sum(x) - c; % Restrição de igualdade h = sum(x) - c
grad_h = ones(size(x)); % Gradiente de h
hessian_h = zeros(size(x)); % Hessiana de h

fo = c'*x;
    
    if k >= max_iter ||  toc(tstart) >= 14400 || isempty(del_L)

        telapsed = toc(tstart);
        fprintf("\nNão convergiu!\n")
        fprintf("Número de iterações: %2.f \n", k);
        fprintf("Função objetivo = %d \n", fo);
        fprintf("Delta L = %d \n", del_L);
        fprintf("Tempo computacional: %d segundos \n", telapsed);

        % Gráfico de Convergência
        figure;
        subplot(2, 1, 1);
        plot(func_values, '-o');
        title('Função Objetivo');
        xlabel('Iterações');
        ylabel('Valor da Função Objetivo');
        
        subplot(2, 1, 2);
        plot(grad_norms, '-o');
        xlabel('Iteração');
        ylabel('Norma do Gradiente da Lagrangiana');
        title('Convergência do Algoritmo');
        grid on;
    end
end

    telapsed = toc(tstart);
    
    fprintf("\nSolução:\n")
    fprintf("Número de iterações: %2.f \n", k);
    fprintf("Função objetivo = %d \n", fo);
    fprintf("Delta L = %d \n", del_L);
    fprintf("Tempo computacional: %d segundos \n", telapsed);

    % Gráfico de Convergência
    figure;
    subplot(2, 1, 1);
    plot(func_values, '-o');
    title('Função Objetivo');
    xlabel('Iterações');
    ylabel('Valor da Função Objetivo');
    
    subplot(2, 1, 2);
    plot(grad_norms, '-o');
    xlabel('Iteração');
    ylabel('Norma do Gradiente da Lagrangiana');
    title('Convergência do Algoritmo');
    grid on;

    return
end