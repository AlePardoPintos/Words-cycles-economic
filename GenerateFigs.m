
cd('/home/alejandro/Paper4/')
%% FIGURE 1

clearvars -except folder
close all

% Initialize figure
figure_num = 1;
figure(figure_num); clf;

% Layout parameters
left_margin = 0.08;
width = 0.36;
horizontal_spacing = 0.06;
vertical_spacing = 0.04;
bottom_margin = 0.085;
top_margin = 0.03;
height_total = 1 - bottom_margin - top_margin - 2 * vertical_spacing;

% Set figure size (comment/uncomment to change)
set(gcf, 'color', 'w', 'position', [1759, 6, 670, 550])

% Initialize axes handles
clear handles
handles(1) = axes('position', [left_margin, bottom_margin + 1*vertical_spacing + 2*height_total/5, width, 3*height_total/5]);
handles(2) = axes('position', [left_margin, bottom_margin + 1*(vertical_spacing + height_total/5), width, height_total/5]);
handles(3) = axes('position', [left_margin, bottom_margin + 0*(vertical_spacing + height_total/5), width, height_total/5]);
handles(4) = axes('position', [left_margin + width + horizontal_spacing, bottom_margin + vertical_spacing + height_total/5, width - horizontal_spacing/2, 4*height_total/5]);
handles(5) = axes('position', [left_margin + width + horizontal_spacing, bottom_margin, width - horizontal_spacing/2, height_total/5]);

font_size = 12;
% Select dataset and load timeline data
load('./Data/Language/TIMELINE_American_3')
year_start_dataset = 1820;

% Time range for analysis
year_start = 1820;
year_end = 2019;
years = year_start:year_end;

% Extract data within selected years
smoothed_data = [TIMELINE.smoothed];
oscillations = smoothed_data((year_start - year_start_dataset + 1):(year_end - year_start_dataset + 1), :);

trends_data = [TIMELINE.trend];
trends = trends_data((year_start - year_start_dataset + 1):(year_end - year_start_dataset + 1), :);

relative_frequency = [TIMELINE.freqrel];
freq_rel = relative_frequency((year_start - year_start_dataset + 1):(year_end - year_start_dataset + 1), :);

word_list = {TIMELINE.word};

% PANEL A: Relative frequency time series
set(gcf, 'currentaxes', handles(1)); hold on

% Words to analyze
selected_words = {'time', 'work', 'god'};
word_indices = find(ismember(word_list, selected_words));

% Custom colors (can be updated)
colors = ones(3) .* [0.3; 0.5; 0.65];
x_text_offset = 1960;
highlight_color = LoadFunc.Colores('American');

% Plot frequency and trend
for i = 1:length(word_indices)
    word = word_list{word_indices(i)};
    plot(years, freq_rel(:, word_indices(i)), ...
        'linewidth', 1.5, ...
        'color', colors(i,:), ...
        'displayname', word)

    trend_plot = plot(years, trends(:, word_indices(i)), ...
        'linewidth', 1.25, ...
        'color', highlight_color);

    nomuestranlegend(trend_plot);
end

ylabel('Relative word usage')
xticklabels({})
xlim([year_start, year_end + 1])
xticks(year_start:40:2020)
ylim([0E-3, 1.75E-3])
yticks((0.4:0.4:2) * 1E-3)
set(gca, 'YGrid', 'off', 'XGrid', 'on', 'fontsize', font_size)
box on

% Add labels near curves
text(1940, 1.57e-3, 'time', 'fontsize', 10)
text(1940, 0.96e-3, 'work', 'fontsize', 10)
text(1940, 0.3e-3, 'god', 'fontsize', 10)

% PANEL B: Oscillations
set(gcf, 'currentaxes', handles(2)); hold on

colors = ones(3) .* [0.4; 0.6; 0.7];
for i = 1:length(word_indices)
    signal = oscillations(:, word_indices(i)) - mean(oscillations(:, word_indices(i)));
    plot(years, signal, 'linewidth', 1.5, ...
        'color', colors(i,:), ...
        'displayname', word_list{word_indices(i)})
end

xticklabels({})
ylabel('Oscillations')
xlim([year_start, year_end + 1])
xticks(year_start:40:2020)
set(gca, 'YGrid', 'off', 'XGrid', 'on', 'fontsize', font_size, 'ytick', [])
ylim([-2, 2] * 1E-4)
box off

% Add vertical borders
y_lims = ylim;
x_lims = xlim;
line([x_lims(2) x_lims(2)], [y_lims(1), y_lims(2)*1.01], 'Color', 'k', 'LineWidth', 0.5)
line([x_lims(1) x_lims(1)], [y_lims(1), y_lims(2)*1.01], 'Color', 'k', 'LineWidth', 0.5)

% PANEL C: GDP growth
set(gcf, 'currentaxes', handles(3))

load('./Data/Econo/SerieParaGranger', 'GrowthGDPFiltrada', 'anios', 'GrowthGDPCruda')
valid_years = anios < 2020;
plot(anios(valid_years), GrowthGDPCruda(valid_years), 'k', 'linewidth', 1.5)

xlim([year_start, year_end + 1])
xticks(year_start:40:2020)
set(gca, 'YGrid', 'off', 'XGrid', 'on', 'fontsize', font_size)
yticklabels({})
ylabel('GDP rate')
xlabel('Year')
box on

% PANEL D: Word spectrum across languages
set(gcf, 'currentaxes', handles(4)); hold on

load('./Data/Language/DataLPCAll_50')
period_range = [1/200, 1/5];
languages_order = {'COHA', 'American', 'British', 'German', 'French', 'Spanish', 'Italian'};
vertical_offsets = [4.3, 2, 0, -1, -2, -3, -4];

for i = 1:length(languages_order)
    lang = languages_order{i};
    color = LoadFunc.Colores(lang);
    mask = ismember({M.base}, lang);

    if isfield(M, 'xLPC')
        freq = M(mask).xLPC;
        power = M(mask).yLPC;
    else
        freq = M(mask).xtrend;
        power = M(mask).ytrend;
    end

    plot(1 ./ freq, power + vertical_offsets(i), ...
        'color', color, ...
        'linewidth', 2.1, ...
        'displayname', lang)
end

set(gca, 'ycolor', 'k', 'fontsize', font_size, ...
         'ytick', [], 'xscale', 'log')
xlim([5, 50])
xticks(sort([50, 30, 18, 12, 9, 6, 5]))
xticklabels({})
ylim([-4, 5.75])
ylabel('Spectral power (arb. units)')
grid on
box on

% Reference vertical lines (periods)
ref_lines = [12, 9];
for val = ref_lines
    pp = line([val val], ylim, 'linestyle', '--', 'color', [0.5 0.5 0.5]);
    nomuestranlegend(pp);
end

legend({}, 'location', 'south', 'Box', 'off', 'EdgeColor', 'none', ...
       'position', [0.84, 0.3, 0.1594, 0.174]);

% PANEL E: GDP spectrum
set(gcf, 'currentaxes', handles(5)); hold on

load('./Data/Econo/SerieParaGranger', 'GrowthGDPCruda', 'anios')
[freqs, spectrum] = HaceLPC(GrowthGDPCruda, 50, 1, 512);
spectrum_normalized = spectrum / 10;

plot(1 ./ freqs, spectrum_normalized + 5.5, ...
    'color', 'k', ...
    'linewidth', 2, ...
    'displayname', 'GDP')

for val = ref_lines
    pp = line([val val], ylim, 'linestyle', '--', 'color', [0.5 0.5 0.5]);
    nomuestranlegend(pp);
end

xlim([5, 50])
xticks(sort([50, 30, 18, 12, 9, 6, 5]))
xlabel('Period (years)')
set(gca, 'ycolor', 'k', 'fontsize', font_size, 'ytick', [], 'xscale', 'log')
grid on
box on
ylim([4.5, 6.6])

% Panel Labels: Add "a", "b", ...
AxesH = axes('Parent', gcf, ...
    'Units', 'normalized', ...
    'Position', [0, 0, 1, 1], ...
    'Visible', 'off', ...
    'XLim', [0, 1], ...
    'YLim', [0, 1], ...
    'NextPlot', 'add');

panel_labels = 'abc';
for panel_idx = 1:2
    if panel_idx == 1
        pos = get(handles(panel_idx), 'position');
    else
        pos = get(handles(panel_idx + 2), 'position');
    end

    if panel_idx == 2
        x_label_pos = pos(1) - 0.025;
    else
        x_label_pos = pos(1) - 0.08;
    end

    y_label_pos = pos(2) + pos(4) + 0.04;
    text(x_label_pos, y_label_pos, panel_labels(panel_idx), 'fontsize', 20)
end

% Save figure
print_pdf('./figs/FiguraIntro2.pdf',gcf)
% exportgraphics(gcf, './figs/FiguraIntro.pdf', 'ContentType', 'vector')


%% Figure 2: Community size distribution and wordclouds
clear all
close all

figure_num = 2;
figure(figure_num); clf
set(gcf, 'color', 'w', 'position', [10, 100, 1374, 757])

clear handles
handles = nan(19, 1);

% Main panel
left_margin = 0.06;
main_width = 0.31;
horizontal_spacing = 0.02;
bottom_margin = 0.1;
main_height = 0.85;
handles(1) = axes('position', [left_margin, bottom_margin, main_width, main_height]);

% Wordcloud + oscillation subpanels
small_width = 0.18;
small_height = 0.23;
start_x = 0.38;
start_y = 0.75;
vertical_spacing = 0.09;
for j = 1:9
    handles(j + 1) = axes('position', ...
        [start_x + mod(j-1,3)*(small_width + horizontal_spacing), ...
         start_y + floor((j-1)/3)*(-small_height - vertical_spacing), ...
         small_width, small_height]);

    handles(j + 10) = axes('position', ...
        [start_x + mod(j-1,3)*(small_width + horizontal_spacing), ...
         start_y + floor((j-1)/3)*(-small_height - vertical_spacing) - small_height*0.28, ...
         small_width, small_height*0.6]);
end

% Main panel: community size distributions
font_size = 16;
gamma = 1.18;
window_width = 199;
corpora = {'COHA', 'British', 'German', 'French', 'Spanish', 'Italian', 'American'};

pp = [];
for i = 1:length(corpora)
    corpus = corpora{i};
    color = LoadFunc.Colores(corpus);

    load(sprintf('./Data/Language/TIMELINE_%s_3', corpus))
    load(sprintf('./Data/Language/Partitions/Particion_%s_3', corpus))

    set(gcf, 'currentaxes', handles(1)); hold on

    Part = SS([SS.ancho] == window_width).todo;
    Part = Part([Part.gamma] == gamma).Particion;

    community_sizes = calcularTamanosComunidadesPorTiempo(Part);
    community_sizes = sort(community_sizes, 'descend');

    pp(end+1) = plot(1:length(community_sizes), community_sizes, '-o', ...
        'linewidth', 2, ...
        'color', color, ...
        'displayname', corpus, ...
        'markerfacecolor', color);

    % Power law fit on top 100
    x_fit = log10(1:length(community_sizes));
    y_fit = log10(community_sizes);
    keep = (1:length(community_sizes)) <= 100;
    x_fit = x_fit(keep);
    y_fit = y_fit(keep);

    params = polyfitn(x_fit, y_fit, 1);
    slope = params.Coefficients(1);
    intercept = params.Coefficients(2);
    r_squared = params.R2;

    fprintf('%s. Slope = %1.3f +/- %1.3f - R² = %1.3f\n', ...
        corpus, slope, params.ParameterStd(2), r_squared)
end

set(gca, 'xscale', 'log', 'yscale', 'log', 'fontsize', font_size)
xlabel('Rank')
ylabel('Community size')
legend(pp([1 7 2:6]), 'location', 'southwest')%, 'Box', 'off', 'EdgeColor', 'none')
legend boxoff
legend({},'Edgecolor','none')
xlim([1 500])
ylim([5 400])
yticks(10.^(0:3))
yticklabels({'', '10^1', '10^2', '10^3'})
grid on
box on

% Wordclouds and oscillations for selected communities
corpus = 'American';
color = LoadFunc.Colores(corpus);
load(sprintf('./Data/Language/TIMELINE_%s_3', corpus))
load(sprintf('./Data/Language/Partitions/Particion_%s_3', corpus))

Part = SS([SS.ancho] == window_width).todo;
Part = Part([Part.gamma] == gamma).Particion;
community_sizes = calcularTamanosComunidadesPorTiempo(Part);
[community_sizes, sorted_indices] = sort(community_sizes, 'descend');

selected_comms = [1 2 3 5 6 13 22 26 37];
max_words = 100;
year_start = 1820;
year_end = 2019;
counter = 1;

for comm_idx = selected_comms
    is_in_comm = Part == sorted_indices(comm_idx);
    
    % Wordcloud panel
    set(gcf, 'currentaxes', handles(counter + 1));
    set(gca, 'Color', 'none')

    words = {TIMELINE(is_in_comm).word};
    freqs = [TIMELINE(is_in_comm).tot] / max([TIMELINE(is_in_comm).tot]);
    wc_colors = repmat([0 0 0], [length(words) 1]);

    h = wordcloud(words, freqs, 'maxdisplaywords', max_words, 'shape', 'oval', 'color', wc_colors);
    handles(counter + 1) = h;
    uistack(handles(counter + 1), 'bottom');

    % Oscillation panel
    set(gcf, 'currentaxes', handles(counter + ceil(length(handles)/2))); hold on
    osc_data = [TIMELINE(is_in_comm).smoothed];
    osc_data = osc_data - mean(osc_data);

    if ismember(comm_idx, [1 3 13 22 26 37])
        % Remove dominant oscillations for clarity
        [~, max_col] = max(max(abs(osc_data)));
        osc_data(:, max_col) = [];
        if comm_idx == 22
            [~, max_col] = max(max(abs(osc_data)));
            osc_data(:, max_col) = [];
        end
    end

    for col = 1:size(osc_data, 2)
        h = plot(year_start:year_end, osc_data(:, col));
        h.Color = [h.Color 0.3];  % Add 30% opacity
    end

    box off
    ylim([-1, 1] * max(abs(ylim)))
    xlim([1819, 2020])
    xticks(1820:40:2020)
    set(gca, 'TickLength', [0.02, 0.02], ...
             'ytick', [], ...
             'ycolor', 'w', ...
             'color', 'none', ...
             'xticklabels', [])

    if comm_idx == 26
        xlabel('Year', 'fontsize', 16);
        xt = get(gca, 'XTick');
        for i = 1:length(xt)
            text(xt(i), ylim(gca) * [1; 0] + 0.23 * range(ylim(gca)), ...
                num2str(xt(i)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'top');
        end
    end

    counter = counter + 1;

    % Highlight selected community in main plot
    set(gcf, 'currentaxes', handles(1));
    pp = scatter(comm_idx, community_sizes(comm_idx), 200, color, 'filled');
    nomuestranlegend(pp);
    text(comm_idx, community_sizes(comm_idx), num2str(comm_idx), ...
        'Color', 'w', ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 10, ...
        'FontWeight', 'bold')
end

% Overlay text labels for wordclouds
AxesH = axes('Parent', gcf, ...
    'Units', 'normalized', ...
    'Position', [0, 0, 1, 1], ...
    'Visible', 'off', ...
    'XLim', [0, 1], ...
    'YLim', [0, 1], ...
    'NextPlot', 'add');

for i = 1:length(selected_comms)
    pos = get(handles(i + ceil(length(handles)/2)), 'position');
    x_pos = pos(1) + 0.01;
    y_pos = pos(2) + 0.28;
    pp = scatter(x_pos, y_pos, 200, LoadFunc.Colores('American'), 'filled');
    nomuestranlegend(pp);
    text(x_pos, y_pos, num2str(selected_comms(i)), ...
        'fontsize', 10, ...
        'fontweight', 'bold', ...
        'color', 'w', ...
        'horizontalalignment', 'center', ...
        'verticalalignment', 'middle');
end

% Export figure
% exportgraphics(gcf, './figs/FiguraComs.pdf', 'ContentType', 'vector')
print_pdf('./figs/FiguraComs2.pdf',gcf)

%% Figure 3: Histogram by corpus and semantic domain
clearvars -except folder
close all

figure_num = 3;
figure(figure_num); clf
clear handles

set(gcf, 'color', 'w', 'position', [1628, 125, 840, 450])

% Layout settings
left_margin = 0.07;
right_margin = 0.02;
bottom_margin = 0.08;
top_margin = 0.07;
horizontal_spacing = 0.08;
width = 1 - left_margin - right_margin - horizontal_spacing;
height = 1 - bottom_margin - top_margin;

% Axes for panels A and B
handles(1) = axes('position', [left_margin, bottom_margin, width/3, height]);
handles(2) = axes('position', [left_margin + width/3 + horizontal_spacing, bottom_margin, 2*width/3, height]);

font_size = 12;
colors = lines(4);
colors(3,:) = colors(4,:);
colors(4,:) = [];

% PANEL A: By corpus
set(gcf, 'currentaxes', handles(1)); hold on
load('./Data/Econo/DataFig3_Bases')

labels = {Results.Base};
LcE   = ([Results.LcE] / length(lags));
Feed  = ([Results.Feed] / length(lags));
EcL   = ([Results.EcL] / length(lags));
None  = ([Results.Nada] / length(lags));
n_communities = sum([Results.Cuantas]);

fprintf('W → E: %1.1f%%\n', sum(LcE) / n_communities * 100)
fprintf('Bidirectional: %1.1f%%\n', sum(Feed) / n_communities * 100)
fprintf('E → W: %1.1f%%\n', sum(EcL) / n_communities * 100)
fprintf('Unrelated: %1.1f%%\n', sum(None) / n_communities * 100)

data = [LcE', Feed', EcL'];

b = bar(data, 'stacked', 'edgecolor', 'none');
b(1).FaceColor = colors(1,:);
b(2).FaceColor = colors(3,:);
b(3).FaceColor = colors(2,:);

bar([Results.Cuantas], 'stacked', 'edgecolor', 'k', 'facecolor', 'none');

xticks(1:length(labels));
for i = 1:length(labels)
    label = labels{i};
    count = Results(i).Cuantas;
    label_with_count = sprintf('%s (%i)', label, count);
    text(i, Results(i).Cuantas + 1, label_with_count, 'rotation', 90, 'fontsize', font_size)
end

xticklabels({});
xlabel('Language')
ylabel('Community count');
xlim([0.5, length(labels) + 0.7])
ylim([0, 43])
set(gca, 'fontsize', font_size)
box on

% PANEL B: By semantic domain
load('./Data/Econo/DataFig3_Campos', 'CampMain', 'lags', 'orden')
set(gcf, 'currentaxes', handles(2)); hold on

[~, order_indices] = ismember(orden, {CampMain.campo});
labels = {CampMain(order_indices).campo};

LcE   = [CampMain(order_indices).LcE] / length(lags) * 100;
Feed  = [CampMain(order_indices).Feed] / length(lags) * 100;
EcL   = [CampMain(order_indices).EcL] / length(lags) * 100;

nada = zeros(size(Feed));
data = [nada', Feed', EcL', LcE'];

b = bar(data, 'stacked', 'edgecolor', 'none');
colors = lines(4);
b(2).FaceColor = colors(4,:);
b(3).FaceColor = colors(2,:);
b(4).FaceColor = colors(1,:);
b(1).FaceColor = [1, 1, 1];
b(1).EdgeColor = [0, 0, 0];

xticks(1:length(labels));
for i = 1:length(labels)
    label = labels{i};
    count = CampMain(strcmp({CampMain.campo}, label)).count;
    label_with_count = sprintf('%s (%i)', label, count);
    text(i, sum(data(i,:)) + 2, label_with_count, 'rotation', 90, 'fontsize', font_size)
end

xticklabels('');
xlabel('Semantic domain')
legend([b(3), b(2), b(4), b(1)], ...
    {'E → W', 'W ↔ E', 'W → E', 'W     E'}, ...
    'location', 'northwest');

ylabel('Lag percentage')
xlim([0.5, length(labels) + 0.7])
ylim([0, 95])
set(gca, 'fontsize', font_size)
box on

% Panel labels ("a", "b")
AxesH = axes('Parent', gcf, ...
    'Units', 'normalized', ...
    'Position', [0, 0, 1, 1], ...
    'Visible', 'off', ...
    'XLim', [0, 1], ...
    'YLim', [0, 1], ...
    'NextPlot', 'add');
labels_ab = 'ab';
for i = 1:length(labels_ab)
    pos = get(handles(i), 'position');
    x_label = pos(1) - 0.055;
    y_label = pos(2) + pos(4) + 0.04;
    text(x_label, y_label, labels_ab(i), 'fontsize', 20)
end

% Export
% exportgraphics(gcf, './figs/FiguraHistBasesSem.pdf', 'ContentType', 'vector')
print_pdf('./figs/FiguraHistBasesSem2.pdf',gcf)

%% Figure 4: Individual fits and Hopf bifurcation
clear all
close all

figure_num = 3;
figure(figure_num); clf
clear handles

set(gcf, 'color', 'w', 'position', [100, 125, 900, 481])

% Axes layout
left_margin = 0.07;
width_main = 0.395;
horizontal_spacing = 0.015;
vertical_spacing = 0.04;
bottom_margin = 0.12;
height_main = 0.7;

handles(1) = axes('position', [left_margin, bottom_margin, width_main, height_main]); % scatter r vs tau
handles(2) = axes('position', [left_margin + width_main, bottom_margin, width_main/4, height_main]); % tau hist
handles(3) = axes('position', [left_margin, bottom_margin + height_main, width_main, height_main/4]); % r hist

% 3 panels for individual fits
hfit_height = (height_main - 2*vertical_spacing)/3;
for i = 1:3
    handles(3 + i) = axes('position', ...
        [left_margin + width_main*5/4 + horizontal_spacing, ...
         bottom_margin + (3 - i)*(hfit_height + vertical_spacing), ...
         width_main, hfit_height]);
end

rmin = 0.1; rmax = 2.5;
taumin = 0.5; taumax = 12;
font_size = 12;
years = 1820:2019;

set(gcf, 'currentaxes', handles(1)); hold on

% Add images (amortiguado & hopf)
image_dir = './figs/AuxImgs/';
axes('position', [.09, .169, .11, .16]);
I = importdata([image_dir 'Imagen1.png']);
h = image(I.cdata);
set(gca, 'xcolor', 'none', 'ycolor', 'none', 'color', 'none')
set(h, 'AlphaData', I.alpha);

axes('position', [.25, .4, .11, .16]);
I = importdata([image_dir 'Imagen3.png']);
h = image(I.cdata);
set(gca, 'xcolor', 'none', 'ycolor', 'none', 'color', 'none')
set(h, 'AlphaData', I.alpha);

% Plot r vs tau scatter and marginals
transparency = 0.3;
selected_corpora = {'COHA', 'German', 'Spanish'};
selected_words = {'body', 'aufmerksamkeit', 'crítico'};

for i = 1:length(selected_corpora)
    corpus = selected_corpora{i};
    word = selected_words{i};
    color = LoadFunc.Colores(corpus);

    load(sprintf('./Data/Language/TIMELINE_%s_3', corpus))
    load(sprintf('./Data/Language/Fits/%sFitsAnalisis2', corpus))

    ratio = [M.RazonSuma];
    Params = [M.ParamsSuma];
    valid = ratio > 0.75 & ratio < 4/3;

    rs_fit = Params(1, valid);
    taus_fit = Params(2, valid);
    M2 = M(valid);

    % Main scatter r vs tau
    set(gcf, 'currentaxes', handles(1))
    scatter(rs_fit, taus_fit, 30, color, 'filled', ...
        'MarkerFaceAlpha', transparency, ...
        'MarkerEdgeAlpha', transparency, ...
        'displayname', corpus)
    xlabel('Growth rate r (1/years)', 'fontsize', font_size)
    ylabel('Delay \tau (years)', 'fontsize', font_size)
    xlim([rmin rmax])
    ylim([taumin taumax])
    xticks(0.5:0.5:rmax)
    yticks(2:2:taumax)
    box on
    legend({},'EdgeColor', 'none')

    % Summary stats
    fprintf('%s r = %1.4f ± %1.4f\n', corpus, mean(rs_fit), std(rs_fit))
    fprintf('%s tau = %1.4f ± %1.4f\n', corpus, mean(taus_fit), std(taus_fit))
    fprintf('%s valid: %1.1f%%\n', corpus, length(M2)/length(M)*100)

    % Tau histogram
    set(gcf, 'currentaxes', handles(2)); hold on
    bins = linspace(taumin, taumax * 1.01, 15);
    [counts, edges] = histcounts(taus_fit, bins);
    edges = edges(2:end) - diff(edges(1:2))/2;
    xq = linspace(edges(1), edges(end), 1000);
    plot(spline(edges, counts, xq), xq, 'color', color, 'linewidth', 2)
    fill(spline(edges, counts, xq), xq, color, 'FaceAlpha', 0.3, 'edgecolor', color)
    set(gca, 'xtick', [], 'ytick', [], 'xcolor', 'w')
    ylim([taumin taumax])

    % r histogram
    set(gcf, 'currentaxes', handles(3)); hold on
    bins = linspace(rmin, rmax * 1.01, 15);
    [counts, edges] = histcounts(rs_fit, bins);
    edges = edges(2:end) - diff(edges(1:2))/2;
    xq = linspace(edges(1), edges(end), 1000);
    plot(xq, spline(edges, counts, xq), 'color', color, 'linewidth', 2)
    area(xq, spline(edges, counts, xq), 'FaceColor', color, 'FaceAlpha', 0.3, 'EdgeColor', color)
    set(gca, 'xtick', [], 'ytick', [], 'ycolor', 'w')
    xlim([rmin rmax])

    % Individual fit plot
    set(gcf, 'currentaxes', handles(3 + i)); hold on
    indw = find(ismember({TIMELINE.word}, word));
    series = TIMELINE(indw).trend + TIMELINE(indw).smoothed;
    plot(years, series, 'k', 'linewidth', 1.5, 'displayname', 'Experimental')

    indm2 = find(ismember({M2.word}, word));
    plot(years, M2(indm2).SolSuma, 'color', color, 'linewidth', 1.5, 'displayname', 'Model')
    xlim([years(1)-1, years(end)+1])
    xticks(1820:40:2020)
    set(gca, 'xgrid', 'on', 'ygrid', 'on', 'fontsize', font_size)
    box on

    if i == 1
        xticklabels({})
        ylim([1.5 6]*1E-4)
        yticks([2 5]*1E-4)
        yticklabels([2 5])
        text(1820, 6.4E-4, 'x10^{-4}', 'fontsize', font_size-1)
    elseif i == 2
        xticklabels({})
        yticks([4 7]*1E-5)
        yticklabels([4 7])
        ylabel('Word usage', 'fontsize', font_size)
        text(1820, 8.42E-5, 'x10^{-5}', 'fontsize', font_size-1)
    else
        ylim([0 3.2]*1E-5)
        yticks([1 3]*1E-5)
        yticklabels([1 3])
        text(1820, 3.5E-5, 'x10^{-5}', 'fontsize', font_size-1)
        xlabel('Year')
    end
end

% Add Hopf bifurcation curves
set(gcf, 'currentaxes', handles(1))
xx = 0:0.01:rmax;
plot(xx, 4 ./ xx, 'k', 'linewidth', 1.5, 'HandleVisibility', 'off');
plot(xx, 8 ./ (27 .* xx), 'k', 'linewidth', 1.5, 'HandleVisibility', 'off');

% Mark selected points
for i = 1:length(selected_corpora)
    corpus = selected_corpora{i};
    word = selected_words{i};
    color = LoadFunc.Colores(corpus);
    load(sprintf('./Data/Language/Fits/%sFitsAnalisis2', corpus))
    ratio = [M.RazonSuma];
    valid = ratio > 0.75 & ratio < 4/3;
    M2 = M(valid);
    indm2 = find(ismember({M2.word}, word));
    params = M2(indm2).ParamsSuma;
    scatter(params(1), params(2), 50, color, 'filled',...
        'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end
set(gca, 'fontsize', font_size)

% Panel labels
AxesH = axes('Parent', gcf, ...
  'Units', 'normalized', ...
  'Position', [0, 0, 1, 1], ...
  'Visible', 'off', ...
  'XLim', [0, 1], ...
  'YLim', [0, 1], ...
  'NextPlot', 'add');

labels = 'ab';
for i = 1:length(labels)
    pos = get(handles(i), 'position');
    if i == 2
        x_pos = pos(1) + 0.075;
    else
        x_pos = pos(1) - 0.03;
    end
    y_pos = pos(2) + pos(4) + 0.06;
    text(x_pos, y_pos, labels(i), 'fontsize', 20)
end

% Export
% exportgraphics(gcf, './figs/FiguraHopf.pdf', 'ContentType', 'vector')
print_pdf('./figs/FiguraHopf2.pdf',gcf)


%% Figure 4: Individual fits and Hopf bifurcation. version 2
clear all
close all

figure_num = 3;
figure(figure_num); clf
clear handles

set(gcf, 'color', 'w', 'position', [100, 125, 900, 481])

% Axes layout
left_margin = 0.07;
width_main = 0.395;
horizontal_spacing = 0.015;
vertical_spacing = 0.04;
bottom_margin = 0.12;
height_main = 0.7;

handles(1) = axes('position', [left_margin, bottom_margin, width_main, height_main]); % scatter r vs tau
handles(2) = axes('position', [left_margin + width_main, bottom_margin, width_main/4, height_main]); % tau hist
handles(3) = axes('position', [left_margin, bottom_margin + height_main, width_main, height_main/4]); % r hist

% 3 panels for individual fits
hfit_height = (height_main - 2*vertical_spacing)/3;
for i = 1:3
    handles(3 + i) = axes('position', ...
        [left_margin + width_main*5/4 + horizontal_spacing, ...
         bottom_margin + (3 - i)*(hfit_height + vertical_spacing), ...
         width_main, hfit_height]);
end

rmin = 0.1; rmax = 2.5;
taumin = 0.5; taumax = 12;
font_size = 12;
years = 1820:2019;

set(gcf, 'currentaxes', handles(1)); hold on

% Add images (amortiguado & hopf)
image_dir = './figs/AuxImgs/';
axes('position', [.09, .169, .11, .16]);
I = importdata([image_dir 'Imagen1.png']);
h = image(I.cdata);
set(gca, 'xcolor', 'none', 'ycolor', 'none', 'color', 'none')
set(h, 'AlphaData', I.alpha);

axes('position', [.25, .4, .11, .16]);
I = importdata([image_dir 'Imagen3.png']);
h = image(I.cdata);
set(gca, 'xcolor', 'none', 'ycolor', 'none', 'color', 'none')
set(h, 'AlphaData', I.alpha);

selected_corpora = {'COHA', 'German', 'Spanish'};
rs_all = [];
taus_all = [];
colores = [];
for i = 1:length(selected_corpora)
    corpus = selected_corpora{i};
    color = LoadFunc.Colores(corpus);

    load(sprintf('./Data/Language/TIMELINE_%s_3', corpus))
    load(sprintf('./Data/Language/Fits/FitsAnalisis%sBest2', corpus))

    Params = [M.params];
    rs_fit = Params(1, :);
    taus_fit = Params(2, :);
    
    near_hopf = taus_fit<=5./rs_fit & taus_fit>=3./rs_fit; 
    Value = [M.valor];
    valid = true(1,length(M));
%     valid = Value>=quantile(Value,0.8);

    rs_all(end+1:end+sum(valid)) = Params(1, valid);
    taus_all(end+1:end+sum(valid)) = Params(2, valid);
    color = repmat(color,[sum(valid) 1]);
    colores = [colores;color];
    
    %Summary stats
    fprintf('%s r = %1.4f ± %1.4f\n', corpus, mean(Params(1, valid)), std(Params(1, valid)))
    fprintf('%s tau = %1.4f ± %1.4f\n', corpus, mean(Params(2, valid)), std(Params(2, valid)))
    fprintf('%s near hopf: %1.1f%%\n', corpus, sum(near_hopf)/length(M)*100)

end

% Plot r vs tau scatter and marginals
transparency = 0.1;
dotsize = 20;
azar = randperm(length(rs_all));
ruido = 0.95+(1.05-0.95).*rand(1,length(rs_all));
% azar = 1:length(rs_fit);

% Main scatter r vs tau
set(gcf, 'currentaxes', handles(1))
scatter(rs_all(azar).*ruido, taus_all(azar).*ruido, dotsize,...
    colores(azar,:), 'filled', ...
    'MarkerFaceAlpha', transparency, ...
    'MarkerEdgeAlpha', transparency, ...
    'HandleVisibility', 'off')
for i = 1:length(selected_corpora)
    corpus = selected_corpora{i};
    color = LoadFunc.Colores(corpus);
    scatter(0, 0, dotsize, color, 'filled', ...
        'displayname', corpus)
end
xlabel('Growth rate r (1/years)', 'fontsize', font_size)
ylabel('Delay \tau (years)', 'fontsize', font_size)
xlim([rmin rmax])
ylim([taumin taumax])
xticks(0.5:0.5:rmax)
yticks(2:2:taumax)
box on
legend({},'EdgeColor', 'none')

% Add Hopf bifurcation curves
xx = 0:0.01:rmax;
plot(xx, 4 ./ xx, 'k', 'linewidth', 1.5, 'HandleVisibility', 'off');
plot(xx, 8 ./ (27 .* xx), 'k', 'linewidth', 1.5, 'HandleVisibility', 'off');


selected_words = {'body', 'aufmerksamkeit', 'crítico'};

for i = 1:length(selected_corpora)
    corpus = selected_corpora{i};
    word = selected_words{i};
    color = LoadFunc.Colores(corpus);

    load(sprintf('./Data/Language/TIMELINE_%s_3', corpus))
    load(sprintf('./Data/Language/Fits/FitsAnalisis%sBest2', corpus))

    Params = [M.params];
    valid = true(1,length(M));

    rs_fit = Params(1, valid);
    taus_fit = Params(2, valid);

    % Tau histogram
    set(gcf, 'currentaxes', handles(2)); hold on
    bins = linspace(taumin, taumax * 1.01, 15);
    [counts, edges] = histcounts(taus_fit, bins);
    edges = edges(2:end) - diff(edges(1:2))/2;
    xq = linspace(edges(1), edges(end), 1000);
    xq = linspace(taumin, taumax, 1000);
    plot(spline(edges, counts, xq), xq, 'color', color, 'linewidth', 2)
    fill(spline(edges, counts, xq), xq, color, 'FaceAlpha', 0.3, 'edgecolor', color)
    set(gca, 'xtick', [], 'ytick', [], 'xcolor', 'w')
    ylim([taumin taumax])
    xlim([30 2080])

    % r histogram
    set(gcf, 'currentaxes', handles(3)); hold on
    bins = linspace(rmin, rmax * 1.01, 15);
    [counts, edges] = histcounts(rs_fit, bins);
    edges = edges(2:end) - diff(edges(1:2))/2;
    xq = linspace(edges(1), edges(end), 1000);
    xq = linspace(rmin, 3, 1000);
    plot(xq, spline(edges, counts, xq), 'color', color, 'linewidth', 2)
    area(xq, spline(edges, counts, xq), 'FaceColor', color, 'FaceAlpha', 0.3, 'EdgeColor', color)
    set(gca, 'xtick', [], 'ytick', [], 'ycolor', 'w')
    xlim([rmin+.06 rmax-.05])

    % Individual fit plot
    set(gcf, 'currentaxes', handles(3 + i)); hold on
    indw = find(ismember({TIMELINE.word}, word));
    series = TIMELINE(indw).trend + TIMELINE(indw).smoothed;
    plot(years, series, 'k', 'linewidth', 1.5, 'displayname', 'Experimental')

    indm = find(ismember({M.word}, word));
    plot(years, M(indm).sol, 'color', color, 'linewidth', 1.5, 'displayname', 'Model')
    xlim([years(1)-1, years(end)+1])
    xticks(1820:40:2020)
    set(gca, 'xgrid', 'on', 'ygrid', 'on', 'fontsize', font_size)
    box on

    if i == 1
        xticklabels({})
        ylim([1.5 6]*1E-4)
        yticks([2 5]*1E-4)
        yticklabels([2 5])
        text(1820, 6.4E-4, 'x10^{-4}', 'fontsize', font_size-1)
    elseif i == 2
        xticklabels({})
        yticks([4 7]*1E-5)
        yticklabels([4 7])
        ylabel('Word usage', 'fontsize', font_size)
        text(1820, 8.42E-5, 'x10^{-5}', 'fontsize', font_size-1)
    else
        ylim([0 3.2]*1E-5)
        yticks([1 3]*1E-5)
        yticklabels([1 3])
        text(1820, 3.5E-5, 'x10^{-5}', 'fontsize', font_size-1)
        xlabel('Year')
    end
end

% Mark selected points
set(gcf, 'currentaxes', handles(1))
for i = 1:length(selected_corpora)
    corpus = selected_corpora{i};
    word = selected_words{i};
    color = LoadFunc.Colores(corpus);
    load(sprintf('./Data/Language/Fits/FitsAnalisis%sBest2', corpus))
    valid = 1:length(M);
    M2 = M(valid);
    indm2 = find(ismember({M2.word}, word));
    params = M2(indm2).params;
    scatter(params(1), params(2), 50, color, 'filled',...
        'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end
set(gca, 'fontsize', font_size)

% Panel labels
AxesH = axes('Parent', gcf, ...
  'Units', 'normalized', ...
  'Position', [0, 0, 1, 1], ...
  'Visible', 'off', ...
  'XLim', [0, 1], ...
  'YLim', [0, 1], ...
  'NextPlot', 'add');

labels = 'ab';
for i = 1:length(labels)
    pos = get(handles(i), 'position');
    if i == 2
        x_pos = pos(1) + 0.075;
    else
        x_pos = pos(1) - 0.03;
    end
    y_pos = pos(2) + pos(4) + 0.06;
    text(x_pos, y_pos, labels(i), 'fontsize', 20)
end

% Export
% exportgraphics(gcf, './figs/FiguraHopf.pdf', 'ContentType', 'vector')
print_pdf('./figs/FiguraHopf2.pdf',gcf)


%% Supplementary Figure: Granger directions as function of lag
clear all
close all

figure_num = 5;
figure(figure_num); clf
clear handles

set(gcf, 'color', 'w', 'position', [100, 125, 780, 417])

% Layout settings
left_margin = 0.07;
right_margin = 0.03;
bottom_margin = 0.12;
top_margin = 0.05;
horizontal_spacing = 0.07;
vertical_spacing = 0.015;

panel_width = (1 - horizontal_spacing - right_margin - left_margin) / 2;
panel_height = 1 - vertical_spacing - bottom_margin - top_margin;

colors = lines(4);
colors(3,:) = colors(4,:);
colors(4,:) = [1 1 1] * 0.7;

handles(1) = axes('position', [left_margin, bottom_margin, panel_width, panel_height]);
handles(2) = axes('position', [left_margin + panel_width + horizontal_spacing, bottom_margin, panel_width, panel_height]);

% Load data
load('./Data/Econo/DataFigSup_Lags', 'cuantos', 'lags', 'leyendas')

% Smoothing settings
smoothing_window = 7;
x_interp = lags(1):0.01:lags(end);
context_labels = {'(country)', '(global)'};
plot_handles = [];

% Loop over panels
for i = 1:2
    for j = 1:3
        y = squeeze(cuantos(i, j, :));
        set(gcf, 'currentaxes', handles(i)); hold on

        mean_smoothed = spline(lags, movmean(y, smoothing_window), x_interp);
        std_smoothed = spline(lags, movstd(y, smoothing_window), x_interp);

        % Plot mean and shaded std
        plot_handles(j) = plot(x_interp, mean_smoothed, ...
            'displayname', leyendas{j}, ...
            'color', colors(j,:), ...
            'linewidth', 1.5);

        pp = fill([x_interp, fliplr(x_interp)], ...
                  [mean_smoothed + std_smoothed, fliplr(mean_smoothed - std_smoothed)], ...
                  colors(j,:), ...
                  'FaceAlpha', 0.2, ...
                  'EdgeColor', 'none');
        nomuestranlegend(pp);
    end

    % Axis formatting
    set(gcf, 'currentaxes', handles(i)); hold on
    title(['GDP ' context_labels{i}], 'fontweight', 'normal')
    xlabel('Lag (years)')
    if i == 1
        ylabel('Proportion')
    end
    if i == 2
        legend([plot_handles(2), plot_handles(3), plot_handles(1)], ...
               {'E → W', 'W ↔ E', 'W → E'}, ...
               'location', 'north');
        legend boxoff
    end
    grid on
    ylim([0 0.6])
    set(gca, 'fontsize', 12)
end

% Panel labels
AxesH = axes('Parent', gcf, ...
    'Units', 'normalized', ...
    'Position', [0, 0, 1, 1], ...
    'Visible', 'off', ...
    'XLim', [0, 1], ...
    'YLim', [0, 1], ...
    'NextPlot', 'add');

labels = 'ab';
for i = 1:length(labels)
    pos = get(handles(i), 'position');
    x_pos = pos(1) - 0.055;
    y_pos = pos(2) + pos(4) + 0.04;
    text(x_pos, y_pos, labels(i), 'fontsize', 20)
end

% Export
print_pdf('./figs/FiguraLagGranger.pdf', gcf)

%% Supplementary Figure: Hopf bifurcation fits for all corpora (new version)
clearvars -except folder
close all

figure_num = 3;
figure(figure_num); clf
clear handles

set(gcf, 'color', 'w', 'position', [100, 125, 900, 481])

% Layout parameters
left_margin = 0.07;
right_margin = 0.03;
bottom_margin = 0.12;
top_margin = 0.055;
horizontal_spacing = 0.05;
vertical_spacing = 0.12;

panel_width = (1 - left_margin - right_margin - 3 * horizontal_spacing) / 4;
panel_height = (1 - top_margin - bottom_margin - vertical_spacing) / 2;

% Top row: 4 panels
for i = 1:4
    handles(i) = axes('position', ...
        [left_margin + (i - 1)*(horizontal_spacing + panel_width), ...
         bottom_margin + vertical_spacing + panel_height, ...
         panel_width, ...
         panel_height]);
end

% Bottom row: 3 panels (1 wider)
for i = 1:3
    w = panel_width;
    if i == 3
        w = panel_width * 1.4;
    end
    handles(i + 4) = axes('position', ...
        [(panel_width + horizontal_spacing)/2 + left_margin + (i - 1)*(horizontal_spacing + panel_width), ...
         bottom_margin, ...
         w, ...
         panel_height]);
end

% Plot settings
rmin = 0.1; rmax = 2.5;
taumin = 0.5; taumax = 12;
font_size = 12;
years = 1820:2019;
xx = 0:0.01:rmax;

transparency = 0.25;
corpora = {'COHA', 'American', 'British', 'German', 'French', 'Spanish', 'Italian'};

for i = 1:length(corpora)
    corpus = corpora{i};
    color = LoadFunc.Colores(corpus);
    
    load(sprintf('./Data/Language/TIMELINE_%s_3', corpus))
    fit_file = sprintf('./Data/Language/Fits/FitsAnalisis%sBest2.mat', corpus);

    set(gcf, 'currentaxes', handles(i)); hold on
    plot(xx, 4 ./ xx, 'k', 'linewidth', 1.5);     % Hopf curve
    plot(xx, 8 ./ (27 .* xx), 'k', 'linewidth', 1.5);  % Secondary bifurcation line

    load(fit_file, 'M')
    params = [M.params];
    fit_errors = [M.valor];
    r_vals = params(1, :);
    tau_vals = params(2, :);
    [~, order] = sort(fit_errors, 'descend');
    
    % Add small noise to avoid overlapping points
    noise = 0.95 + (1.05 - 0.95) * rand(size(r_vals));
    
    scatter(r_vals(order).*noise, tau_vals(order).*noise, 10, fit_errors(order), ...
        'filled', 'MarkerFaceAlpha', transparency, 'MarkerEdgeAlpha', transparency)

    caxis([-4.5 -3.5])
    xlim([rmin rmax])
    ylim([taumin taumax])
    xticks(0.5:0.5:rmax)
    yticks(2:2:taumax)
    box on
    set(gca, 'fontsize', font_size)
    title(corpus, 'fontweight', 'normal')

    if i == 6
        xlabel('Growth rate r (1/years)', 'fontsize', font_size)
    end
    if i == 1 || i == 5
        ylabel('Delay \tau (years)', 'fontsize', font_size)
    end
    if i == 7
        cb = colorbar;
        ylabel(cb, 'Fitting error');
    end
end

% Export
% print_pdf('./figs/FiguraSupHopfNuevosAjustes.pdf', gcf)
