function AAMOEA(Global)
% <algorithm> <A>
% An adaptive modeling-based multi-objective evolutionary algorithm 
% Multi-/Many-objective Optimiation
% gmax  ---  20 --- Number of generations before updating Kriging Kmodels

%------------------------------- Reference --------------------------------

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% reseArch purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by ShufenQin

    %% Parameter setting
    [gmax] = Global.ParameterSet(20);
    
    %% Initialization
    [V,~] = UniformPoint(Global.N,Global.M);

    NI    = 11*Global.D-1;%100;%
    Plhs  = lhsamp(NI,Global.D);
    Arc    = INDIVIDUAL(repmat(Global.upper-Global.lower,NI,1) .* Plhs + repmat(Global.lower,NI,1));
    
    lob    = 1e-5.*ones(1,Global.D);
    upb    = 100.*ones(1,Global.D);
    THETA  = 5.*ones(Global.M,Global.D); 

    %% Optimization
    Plhsp  = lhsamp(Global.N,Global.D);
    PopDec  =  repmat(Global.upper-Global.lower,Global.N,1) .* Plhsp + repmat(Global.lower,Global.N,1); 
 
    t = 1;
    randlist = randperm(length(Arc));
    Tr = Arc(randlist(1:0.7*length(Arc)));
    Ts = Arc(randlist(0.7*length(Arc)+1:end));
 
    while Global.NotTermination(Arc)  
        %% The first training 
        TrDec = Tr.decs;
        TrObj = Tr.objs;
        if t==1  
            for j = 1 : Global.M 
                dmodel  = dacefit(TrDec,TrObj(:,j),'regpoly0','corrgauss',THETA(j,:),lob,upb);
                Kmodel{t,j}  = dmodel;
                THETA(j,:) = dmodel.theta;
                %prediction
                  predf = [];deTs=[];
                for i = 1:length(Ts)
                    [predf(i,j),~,deTs(i,j)] = predictor(Ts(i).decs,Kmodel{t,j});
                end
            end
            sigma(t,:) = sqrt(sum(((predf-Ts.objs)./(max(Ts.objs,[],1)-min(Ts.objs,[],1))).^2,1)./(length(Ts)-1))';
          
        else 
            % The third training
            predf = [];deTs=[];
            for j = 1:Global.M
                for i = 1:length(Ts)
                    [predf(i,j),~,deTs(i,j)] = predictor(Ts(i).decs,Kmodel{t-1,j});
                end
            end
            sigma(t,:) = sqrt(sum(((predf-Ts.objs)./(max(Ts.objs,[],1)-min(Ts.objs,[],1))).^2,1)./(length(Ts)-1))';
            flag{t,1} =  sigma(t,:)>sigma(t-1,:);
            for j = 1:Global.M
                if flag{t,1}(1,j)==1
                    dmodel  = dacefit(TrDec,TrObj(:,j),'regpoly0','corrgauss',THETA(j,:),lob,upb);
                    Kmodel{t,j}  = dmodel;
                    THETA(j,:) = dmodel.theta;
                else
                    Kmodel{t,j}  =  Kmodel{t-1,j};
                    THETA(j,:) = Kmodel{t-1,j}.theta;
                end
            end
        end
       
        g = 1;
        while g <= gmax
            drawnow();    
            %% 基于目标函数优化
            OffDec = GA(PopDec);
            OffDec = unique(OffDec,'rows');
            PopDect = [PopDec;OffDec];
            N  = size(PopDect,1);
            PopObjtp = zeros(N,Global.M);
            PopObjt = zeros(N,Global.M);
            detat    = zeros(N,Global.M); 
         
            for j = 1:Global.M
                for i = 1:N
                    [PopObjt(i,j),~,detat(i,j)] = predictor(PopDect(i,:),Kmodel{t,j});
                end
            end
            OffObj = PopObjt(N-size(OffDec,1)+1:end,:);
 
            indexf  = KrigingSelection_tran(PopObjt,V);
            PopDec = PopDect(indexf,:); 
            
            mes = sqrt(detat);
            so = mes(N-size(OffDec,1)+1:end,:);
            g = g + 1; 
        end 
     %% Infill sampling
      offArc = [OffDec;Arc.decs];
      offArcf = [OffObj;Arc.objs];
      fronum = NDSort(offArcf,1)==1;
      fno = find(fronum(1:size(offArc)-length(Arc)));
      if ~isempty(fno)
          Offsel = offArc(fno,:);
          Offself = offArcf(fno,:); 
          disArcf = pdist2(Offself,[Offself;Arc.objs],'cosine');
          [disAsf,indsf] = sort(disArcf,2); 
          [~,id] = sort(disAsf(:,2));
          Popreal = Offsel(id(end),:); 
      else 
          if isempty(OffDec)
              Popreal = [];
          else
              disArcf1 = pdist2(OffObj,[OffObj;Arc.objs],'cosine');
              [disAsf1,indsf] = sort(disArcf1,2);
              [~,idp] = sort(disAsf1(:,2));
              Popreal = OffDec(idp(end),:);
          end 
      end
      

        [soSort,~] = sort(so,2,'ascend');
        fobj = soSort(:,[1,end]);
        OffNd = OffDec(NDSort([fobj(:,1),-fobj(:,2)],1)==1,:); 
          if size(OffNd,1)>1 

              soNd = soSort(NDSort([fobj(:,1),-fobj(:,2)],1)==1,:);
              [~,sorid] = sort(mean(soNd(:,2:end-1),2));
              Popreal = [Popreal;OffNd(sorid(end),:)];
          else
              Popreal= [Popreal;OffNd]; 
          end
      
         
        % Delete duplicated solutions
        Poprealu = unique(Popreal,'rows');
        for i = 1: size(Poprealu,1)
            [~,index] = unique([Arc.decs;Poprealu(i,:)],'rows');
            if length(index) == size([Arc.decs;Poprealu(i,:)],1)
                PopNew = INDIVIDUAL(Poprealu(i,:));
                Arc     = [Arc,PopNew];
                Tr     = [Tr,PopNew];
                Ts     = [Ts,PopNew];
            end 
        end
         
        t = t+1;
       %% The next population    
        if size(PopDec,1)<Global.N
            Arcnd = Arc((NDSort(Arc.objs,1)==1));
            if length(Arcnd)<Global.N-size(PopDec,1)
                Arcnon = Arc((NDSort(Arc.objs,1)~=1));
                randlist = randperm(length(Arcnon));
                Arcr = Arcnon(randlist(1:Global.N-size(PopDec,1)-length(Arcnd)));
                PopDec = [PopDec;Arcnd.decs;Arcr.decs];
            else
                randlist = randperm(length(Arcnd));
                Poptemp = Arcnd(randlist(1:Global.N-size(PopDec,1)));
                PopDec = [PopDec;Poptemp.decs];
            end
        end
      
    end
end