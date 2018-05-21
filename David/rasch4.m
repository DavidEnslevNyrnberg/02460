% LK Hansen, DTU Compute (c) 2018
% NOT FOR DISTRIBUTION - Not final version
%
% Rasch model simulation and learning based on Sufficient Stat perturbation
% Dwork, C. and Smith, A., 2010. 
% Differential privacy for statistics: What we know and what we want to learn. 
% Journal of Privacy and Confidentiality, 1(2), p.2.
%
% Number of tests
D=100; 
% Number of students
Ns=10;
Nmax=200;
Nmin=10;
Narr=round(linspace(Nmin,Nmax,Ns));
% privacy budget
epsi=10.0;
% distribution of students
mu0=0;
sig0=1;
% distribution of task easyness
scal0=1;
lowlev=0.5;
%
lam=10;   % L2 regularization - should be increased as epsi -> 0
%
Rep=50;  % repetitions
for n=1:Ns,
    N=Narr(n);
    for r=1:Rep,
        bet0=mu0+sig0*randn(N,1);
        del0=lowlev+scal0*randn(1,D);
        p0=1./(1 + exp(-bet0*ones(1,D) -ones(N,1)*del0)); % p(x=1|bet0,delta0)
        disp(['N = ',int2str(N),', R = ',int2str(r)])
        X=double(rand(N,D)<p0);   % generate data
        Xtest=double(rand(N,D)<p0); % test set
        % sufficient stats
        xd=sum(X,2);
        xn=sum(X,1);
        % initialization
        bet=0.001*randn(N,1);
        del=0.001*randn(1,D);
        %     
        w0=[bet;del'];
        options = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'MaxIter',10^5,'MaxFunEvals',10^5,'TolX',10^-5);
        % tune parameters
        w=fminunc(@(w) rasch_negloglik_grad_priv_suff(w,xd,xn,lam),w0,options);
        bet=w(1:N);
        del=w((N+1):(N+D))';
        pe=1./(1 + exp(-bet*ones(1,D) -ones(N,1)*del));  % p(x=1|bet,delta)
        % private suff stats
        xd_priv=max(zeros(size(xd)),xd+(D/epsi)*sign(randn(size(xd))).*log(rand(size(xd))));
        xd_priv=min(D*ones(size(xd_priv)),xd_priv);
        xn_priv=max(zeros(size(xn)),xn+(D/epsi)*sign(randn(size(xn))).*log(rand(size(xn))));
        xn_priv=min(N*ones(size(xn_priv)),xn_priv);
        % optimize with private suff stats
        w_priv=fminunc(@(w) rasch_negloglik_grad_priv_suff(w,xd_priv,xn_priv,lam),w0,options);
        bet_priv=w(1:N);  % when we compute probabilities this step approximates that each student computes her grade on her own non-private beta
        del_priv=w_priv((N+1):(N+D))';
        pe_priv=1./(1 + exp(-bet_priv*ones(1,D) -ones(N,1)*del_priv));
        T=double(pe>0.5); % map estimate
        T0=double(p0>0.5);
        Etest(n,r)=(1-mean(mean(T==Xtest)));
        Etest0(n,r)=(1-mean(mean(T0==Xtest)));
        T_priv=double(pe_priv>0.5);
        Etest_priv(n,r)=(1-mean(mean(T_priv==Xtest)));
        Edel(n,r)=norm(del-del0)/norm(del0);
        Edel_priv(n,r)=norm(del_priv-del0)/norm(del0);
        Ebet(n,r)=norm(bet-bet0)/norm(bet0);
        a=corrcoef(pe,p0);
        p0pecorr(n,r)=a(1,2);
        a=corrcoef(pe_priv,p0);
        p0pe_privcorr(n,r)=a(1,2);
end
end
figure(1), 
subplot(2,2,1), plot(Narr,mean(Edel,2),'r',Narr,mean(Edel,2)+(1/sqrt(Rep))*std(Edel,[],2),':r',...
    Narr,median(Edel_priv,2),'g',Narr,median(Edel_priv,2)+(1/sqrt(Rep))*std(Edel_priv,[],2),':g',...
    Narr,median(Edel_priv,2)-(1/sqrt(Rep))*std(Edel_priv,[],2),':g',...
    Narr,mean(Edel,2)-(1/sqrt(Rep))*std(Edel,[],2),':r',Narr,mean(Ebet,2),'b',Narr,mean(Ebet,2)+(1/sqrt(Rep))*std(Ebet,[],2),':b',...
    Narr,mean(Ebet,2)-(1/sqrt(Rep))*std(Ebet,[],2),':b'), xlabel('N'),ylabel('NMSE'), title(['ERROR IN BETA & DELTA, D =',int2str(D)]),grid
axis([Nmin, Nmax,0.0,5.0]),
subplot(2,2,2), plot(Narr,mean(Etest,2),'r',Narr,mean(Etest0,2),'b',Narr,mean(Etest_priv,2),'g'), xlabel('N'),ylabel('NMSE')
grid,axis([Nmin, Nmax,0.0,2.0]),legend('NON','ORACLE','PRIV'),legend('NON','ORACLE','PRIV'),title('TEST ERROR / ORACLE TEST ERROR')
subplot(2,2,3), plot(Narr,mean(p0pecorr,2),'r',Narr,mean(p0pe_privcorr,2),'b'), xlabel('N'),ylabel('p p0 CORR')
axis([Nmin, Nmax,0.0,1.0]),legend('NON','PRIV'), title('PROB CORRELATION W/ GROUND TRUTH'), grid

figure, subplot(2,1,1),bar(del),title('DEL')
subplot(2,1,2),bar(del_priv), title('PRIVATE DEL')
 
figure, subplot(2,1,1),plot(p0(:),pe_priv(:),'b.'),title('EST PRIVATE PROBABILITY VS GROUND TRUTH')
subplot(2,1,2),plot(p0(:),pe(:),'b.'),title('EST  PROBABILITY VS GROUND TRUTH')
xlabel('GROUND TRUTH')










