%% Post processing for sparse T2V by FPINN framework
% Evaluate reconstruction performance
clc
clear all
close all

% prepare folders
addpath('.\Functions');
mkdir(strcat('Figures\FPINN_paperDisplay'));
% mkdir(strcat('Figures\',date));
mkdir(strcat('Data\FPINN'));
dataLoc   = num2str(pwd);
% Load reference data
file_delete = ['postProcessing'];


%% data post-processing
case_name   = {'ref';'_case1';'_case2';'_case3'};
keySet      = {'U';'V';'P';'T';'Ret_rev';'hist'};

% Load reference and prediction data
for k = 1:numel(case_name)
    % Define a container
    pred{k} = containers.Map('KeyType','char','ValueType','any');
    if k == 1 % Load reference case
        file_name   = sprintf('%sFPINN\\TrainData\\20230627_PlaneJet2D_Benchmark_nonDimensonal.mat',...
        dataLoc(1:end-numel(file_delete)));
        ref=load(file_name);
        xmesh = ref.xmesh_;
        ymesh = ref.ymesh_;
        % Load 5 variables and save in 'pred' container for easy display
        pred{k}('U') = ref.U_{1};
        pred{k}('V') = ref.V_{1};
        pred{k}('P') = ref.P_{1};
        pred{k}('T') = ref.T_{1};
        pred{k}('Ret_rev') = ref.Ret_rev{1};
    else
        file_name = sprintf('%sFPINN\\%s\\pred\\PINN_results.mat',...
            dataLoc(1:end-numel(file_delete)),case_name{k});
        fdata=load(file_name);
        
        for i = 1:5 % 'U V P T Re_rev'
            pred{k}(keySet{i})= reshape(fdata.pred(:,i),size(xmesh'))';
        end
        hist = [];
        if ismember(k,[3,4])
            hist(:,1)    = fdata.hist(:,1)+fdata.hist(:,2);   % Combine data (internal info) + bc -> data
            hist(:,2:5)  = fdata.hist(:,3:6);
        else
            hist    = fdata.hist;
        end
        pred{k}('hist')=hist;
    end
end

%% Paper display
plotParameters;
%--------------------------------------------------------------------------
% Train history 
%--------------------------------------------------------------------------
for k = 2:numel(case_name)
    FigHandle = figure; hold on
    set(FigHandle, 'Position', [150, 80, 600, 450]);
    yp = pred{k}('hist');
    for i = [2 3 4 5 1]
        plot(yp(:,i),'color',co(i,:),'lineWidth',1.5);
    end

   % Legend
    legend({'$\lambda_{1} e_{1}$','$\lambda_{1} e_{2}$',...
        '$\lambda_{1} e_{3}$','$\lambda_{1} e_{4}$','$\lambda_{5} L_{data}$'},...
        'Location','best','interpreter','latex','fontsize',18,'fontname','times');
    legend('boxoff');

    set(gca,'yscale','log');
    set(gca,'xlim',[0 3.2e4]);
    set(gca,'ylim',[1e-8 1]);
    ylabel('$Loss$','interpreter','latex');
    xlabel('$Epoch$','interpreter','latex');
    set(gca,'fontsize',18,'fontname','times'); 
    exportgraphics(FigHandle,strcat(['Figures\FPINN_paperDisplay\' ...
        'trainHistory'],case_name{k},'.png'),'Resolution',450);
end
close all

%--------------------------------------------------------------------------
% Contours comparsion
%--------------------------------------------------------------------------
variable_display = {'\overline{U}','\overline{V}','\overline{P}','\overline{T}','1/Re_{t}'};
cmax_error = [0.2 0.02 1e-3 0.015 0.08];
for var = 1:numel(keySet)-1
    case_lgd = [{sprintf('$ref\\ %s$',variable_display{var})};...
        {'$case1$'};{'$case2$'};{'$case3$'};{'$|ref-case3|$'}];
    
    % field plot
    FigHandle = figure; hold on
    set(FigHandle, 'Position', [150, 80, 2200, 300]);
    t = tiledlayout(1,5,'TileSpacing','Compact','Padding','Compact');    
    
    for k = 1:5 % ref, case1, case2, case3, ref-case3
        if ismember(k,[1])
            field = pred{k}(keySet{var});
            cmin  = min(field(:));
            cmax  = max(field(:));
        elseif ismember(k,[5])
            field = abs(pred{4}(keySet{var})-pred{1}(keySet{var}));
            cmax  = max(field(:));
        elseif ismember(k,[2 3 4])
            field = pred{k}(keySet{var});
        end
    
        nexttile 
        
        pcolor(xmesh,ymesh,field); hold on
        set(gca,'xlim',[0 125]);
        set(gca,'ylim',[-30 30]);
    
        shading interp;  
        colormap jet;
    
        if k == 1
            ylabel('$x/d$','interpreter','latex');
            xlabel('$y/d$','interpreter','latex');
            set(gca,'xtick',[0:25:125]);
            set(gca,'ytick',[-30:15:30]);
        else
            set(gca,'xtick',[],'ytick',[]);
        end
    
        if ismember(k,[1 2 3 4])
            caxis([0.85*cmin 0.85*cmax]); % for better display
        else
            caxis([0 cmax_error(var)]); % for better display
        end
    
        title (case_lgd{k},'interpreter','latex','fontsize',15,'fontname','times');
        set(gca,'fontsize',16,'fontname','times'); 
    
        if ismember(k,[1,5])
            colorbar;
        end

        % windows of feeded U and V as known info
        if ismember(k,[3,4]) && ismember(var,[1 2])
            if ismember(k,[4])
                rectangle('Position',[52.9555 -4.5625 40 4.8125],'EdgeColor','r','linewidth',2);
            end
            % Inlet BC
            plot([0,0],[-40,40],'color','r','linewidth',3);
        end 

        % windows of feeded T
        if ismember(var,[4]) && ismember(k,[2,3,4])
            rectangle('Position',[0 -350 125 70],'EdgeColor','r','linewidth',3.5);
        end 
        axis equal
    end
    
    exportgraphics(FigHandle,sprintf(['Figures\\FPINN_paperDisplay\\' ...
        'contours_comparsion_%s.png'],keySet{var}),'Resolution',900);
end

%--------------------------------------------------------------------------
% Profile comparsion -> y= 0 m centerline
%--------------------------------------------------------------------------
% centerline comparsion
col_plot = [rgb('black');rgb('green');rgb('red')];

idx_cen = 48;
FigHandle = figure; hold on
set(FigHandle, 'Position', [150, 80, 600, 450]);
for k = [2 3 4 1]
    
%     variabe_field = sqrt(pred{k}('U').^2+pred{k}('V').^2);
    variabe_field  = pred{k}('U');
    xp    = xmesh(1,:); 
    yp    = variabe_field(idx_cen,:);
    if k == 1
        h(k)  = plot(xp,yp,'--','color','b','linewidth',2.5); 
    else
        h(k)  = plot(xp,yp,'-','color',col_plot(k-1,:),'linewidth',2); 
    end
end

% Legend
legend({'Case1';'Case2';'Case3';'refernce'},...
    'Location','best','interpreter','latex','fontsize',18,'fontname','times');
legend('boxoff');

set(gca,'xlim',[0 150]);
set(gca,'ylim',[-0.1 1.1]);
xlabel('$x/d$','interpreter','latex');
ylabel('$\overline{U}$','interpreter','latex');
set(gca,'fontsize',18,'fontname','times'); 
exportgraphics(FigHandle,sprintf(['Figures\\FPINN_paperDisplay\\' ...
        'centerline_comparsion_U.png']),'Resolution',450);

%--------------------------------------------------------------------------
% Profile comparsion -> x = 0.5m slice
%--------------------------------------------------------------------------
slice_display = 0.5; % x= 0.5m
d             = 16e-3;
[~,idx_slice] = min(abs(xmesh(1,:)-slice_display/d));

FigHandle = figure; hold on
set(FigHandle, 'Position', [150, 80, 600, 450]);
for k = [2 3 4 1]
    
%     variabe_field = sqrt(pred{k}('U').^2+pred{k}('V').^2);
    variabe_field  = pred{k}('U');
    yp    = ymesh(:,1); 
    xp    = variabe_field(:,idx_slice);
    if k == 1
        h(k)  = plot(xp,yp,'--','color','b','linewidth',2.5); 
    else
        h(k)  = plot(xp,yp,'-','color',col_plot(k-1,:),'linewidth',2); 
    end
end

% Legend
legend({'Case1';'Case2';'Case3';'refernce'},...
    'Location','northeast','interpreter','latex','fontsize',18,'fontname','times');
legend('boxoff');

set(gca,'xlim',[-0.15 0.54]);
set(gca,'ylim',[-15 15]);
ylabel('$y/d$','interpreter','latex');
xlabel('$\overline{U}$','interpreter','latex');
set(gca,'fontsize',18,'fontname','times'); 
exportgraphics(FigHandle,sprintf(['Figures\\FPINN_paperDisplay\\' ...
        'slice_comparsion_U.png']),'Resolution',450);