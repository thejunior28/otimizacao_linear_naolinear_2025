function [x, fo,i] = armijo(f, df, x0, alpha, beta, sigma, n_max)

x = x0; % Ponto inicial
    fo = f(x);
    i=0; %iterações
    for k = 1:n_max
        grad = df(x);
        if df(x) <= 10^-6
       fprintf("Derivada muito próxima de zero!\n")
       fprintf("Iteração %2.f\n",i)
        end
        % Condição de Armijo
        while f(x - alpha * grad) > fo - sigma * alpha * norm(grad)^2
            alpha = beta * alpha; % Redução do peso
        end
        
        % Atualização
        x = x - alpha * grad;
        fo = f(x);
        i=i+1;

        % Condição de parada
        if norm(grad) < 1e-6
            break;
        end
    end
end