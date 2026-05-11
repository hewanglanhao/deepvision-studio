package com.deepvision.studio.training;

import com.deepvision.studio.training.TrainingDtos.DatasetImportResponse;
import com.deepvision.studio.training.TrainingDtos.ImagePreviewItem;
import com.deepvision.studio.training.TrainingDtos.LabelDistributionItem;
import com.deepvision.studio.training.TrainingDtos.PointPreviewItem;
import com.deepvision.studio.training.TrainingDtos.TablePreview;
import com.deepvision.studio.training.TrainingDtos.TrainingDatasetDetail;
import com.deepvision.studio.training.TrainingDtos.TrainingDatasetOption;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import javax.imageio.ImageIO;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

@Service
public class TrainingDatasetService {
  private static final List<String> COLORS = List.of(
      "#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed",
      "#0891b2", "#ca8a04", "#db2777", "#4b5563", "#65a30d"
  );
  private static final Set<String> IMAGE_EXTENSIONS = Set.of("png", "jpg", "jpeg", "gif", "bmp");
  private static final DateTimeFormatter UPLOAD_ID_TIME = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss");
  private static final long MAX_UPLOAD_BYTES = 200L * 1024L * 1024L;
  private static final int MAX_IMAGE_COUNT = 5000;
  private static final int MIN_CLASS_COUNT = 2;
  private static final int MIN_ROWS = 10;
  private static final int MIN_SAMPLES_PER_IMAGE_CLASS = 2;
  private static final int PREVIEW_IMAGES_PER_CLASS = 3;

  private final TrainingDatasetRepository datasets;
  private final ObjectMapper objectMapper;
  private final Path uploadsRoot;
  private final Path datasetsRoot;

  public TrainingDatasetService(
      TrainingDatasetRepository datasets,
      ObjectMapper objectMapper,
      @Value("${deepvision.uploads.root}") String uploadsRoot,
      @Value("${deepvision.datasets.root}") String datasetsRoot
  ) {
    this.datasets = datasets;
    this.objectMapper = objectMapper;
    this.uploadsRoot = Path.of(uploadsRoot).toAbsolutePath().normalize();
    this.datasetsRoot = Path.of(datasetsRoot).toAbsolutePath().normalize();
  }

  @PostConstruct
  void init() {
    registerBuiltinDatasets();
  }

  public List<TrainingDatasetOption> listDatasets(String source) {
    List<TrainingDataset> rows = source == null || source.isBlank()
        ? datasets.findAllByOrderBySourceAscNameAsc()
        : datasets.findBySourceOrderByNameAsc(source);
    return rows.stream()
        .map(this::toDetail)
        .map(TrainingDatasetDetail::toOption)
        .toList();
  }

  public List<TrainingDatasetOption> listBuiltin() {
    return listDatasets("builtin");
  }

  public TrainingDatasetDetail getDetail(String datasetId) {
    return datasets.findById(datasetId)
        .map(this::toDetail)
        .orElseThrow(() -> new IllegalArgumentException("Dataset not found."));
  }

  public boolean exists(String datasetId) {
    return datasets.existsById(datasetId);
  }

  public void deleteUploadedDataset(String datasetId) {
    TrainingDataset row = datasets.findById(datasetId)
        .orElseThrow(() -> new IllegalArgumentException("Dataset not found."));
    if (!"upload".equals(row.getSource())) {
      throw new IllegalArgumentException("Built-in datasets cannot be deleted.");
    }
    Path datasetDir = uploadDatasetDir(datasetId);
    try {
      deleteDirectoryIfExists(datasetDir);
      datasets.delete(row);
    } catch (IOException ex) {
      throw new IllegalArgumentException("Failed to delete uploaded dataset files.");
    }
  }

  public DatasetImportResponse importDataset(MultipartFile[] files, String labelColumn, Integer requestedClassCount) {
    if (files == null || files.length == 0) {
      throw new IllegalArgumentException("No dataset files uploaded.");
    }
    List<MultipartFile> nonEmpty = List.of(files).stream()
        .filter(file -> file != null && !file.isEmpty())
        .toList();
    if (nonEmpty.isEmpty()) {
      throw new IllegalArgumentException("No dataset files uploaded.");
    }

    long totalBytes = nonEmpty.stream().mapToLong(MultipartFile::getSize).sum();
    if (totalBytes > MAX_UPLOAD_BYTES) {
      throw new IllegalArgumentException("Dataset upload is too large. Please keep it under 200MB.");
    }

    List<MultipartFile> zipFiles = nonEmpty.stream().filter(this::isZipFile).toList();
    List<MultipartFile> csvFiles = nonEmpty.stream().filter(this::isCsvFile).toList();
    List<MultipartFile> imageFiles = nonEmpty.stream().filter(this::isImageFile).toList();
    if (!zipFiles.isEmpty() && nonEmpty.size() > zipFiles.size()) {
      throw new IllegalArgumentException("Please upload either one ZIP file, one CSV file, or image files, not mixed files.");
    }
    if (zipFiles.size() > 1) {
      throw new IllegalArgumentException("Only one ZIP dataset is supported per import.");
    }
    if (!csvFiles.isEmpty() && !imageFiles.isEmpty()) {
      throw new IllegalArgumentException("Please upload either one CSV file or image files, not both.");
    }
    if (csvFiles.size() > 1) {
      throw new IllegalArgumentException("Only one CSV file is supported per import.");
    }
    if (zipFiles.isEmpty() && csvFiles.isEmpty() && imageFiles.isEmpty()) {
      throw new IllegalArgumentException("Only ZIP, CSV, or image files are supported.");
    }

    TrainingDatasetDetail detail = zipFiles.size() == 1
        ? importImageZip(zipFiles.get(0))
        : csvFiles.size() == 1
        ? importCsv(csvFiles.get(0), labelColumn, requestedClassCount)
        : importImages(imageFiles);
    datasets.save(toEntity(detail));
    return new DatasetImportResponse(detail.id(), detail);
  }

  String builtInPreviewSvg(String datasetId, int index) {
    TrainingDatasetDetail detail = getDetail(datasetId);
    if (!"builtin".equals(detail.source()) || !"image".equals(detail.kind())) {
      throw new IllegalArgumentException("Preview not found.");
    }
    int safeIndex = Math.max(0, index - 1);
    List<String> labels = detail.labels();
    int labelIndex = (safeIndex / PREVIEW_IMAGES_PER_CLASS) % labels.size();
    String label = labels.get(labelIndex);
    String foreground = "mnist-1000".equals(datasetId) ? "#111827" : COLORS.get(labelIndex % COLORS.size());
    String background = "mnist-1000".equals(datasetId) ? "#f8fafc" : "#e0f2fe";
    String text = "mnist-1000".equals(datasetId) ? label : label.substring(0, Math.min(2, label.length())).toUpperCase(Locale.ROOT);
    return """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">
          <rect width="96" height="96" rx="14" fill="%s"/>
          <text x="48" y="56" font-family="Arial, sans-serif" font-size="30" text-anchor="middle" fill="%s" font-weight="700">%s</text>
        </svg>
        """.formatted(background, foreground, escapeXml(text));
  }

  private TrainingDatasetDetail importCsv(MultipartFile file, String labelColumn, Integer requestedClassCount) {
    String datasetId = nextUploadId();
    Path datasetDir = uploadDatasetDir(datasetId);
    Path csvTarget = datasetDir.resolve("data.csv").normalize();
    if (!csvTarget.startsWith(datasetDir)) {
      throw new IllegalArgumentException("Invalid CSV path.");
    }

    byte[] bytes;
    List<String> lines;
    try {
      bytes = file.getBytes();
      String text = new String(bytes, StandardCharsets.UTF_8);
      lines = text.lines().map(String::trim).filter(line -> !line.isBlank()).toList();
    } catch (IOException ex) {
      throw new IllegalArgumentException("Failed to read CSV file.");
    }
    if (lines.size() < 2) {
      throw new IllegalArgumentException("CSV requires a header and at least one data row.");
    }

    List<String> headers = parseCsvLine(lines.get(0));
    List<List<String>> rows = lines.stream().skip(1).map(this::parseCsvLine).filter(row -> !row.isEmpty()).toList();
    if (rows.size() < MIN_ROWS) {
      throw new IllegalArgumentException("CSV dataset requires at least " + MIN_ROWS + " data rows.");
    }
    long malformedRows = rows.stream().filter(row -> row.size() != headers.size()).count();
    if (malformedRows > 0) {
      throw new IllegalArgumentException("CSV row column count must match the header. Found " + malformedRows + " malformed rows.");
    }
    int labelIndex = resolveRequiredLabelColumn(headers, labelColumn);
    if (labelIndex < 0) {
      throw new IllegalArgumentException("CSV label column is required.");
    }
    int classCount = resolveRequiredClassCount(requestedClassCount);
    Map<String, Integer> labelCounts = new LinkedHashMap<>();
    int missingValues = 0;
    int blankLabelRows = 0;
    Set<Integer> categoricalColumns = new java.util.LinkedHashSet<>();
    for (List<String> row : rows) {
      for (int i = 0; i < headers.size(); i += 1) {
        if (i >= row.size() || row.get(i).isBlank()) {
          missingValues += 1;
          continue;
        }
        if (i != labelIndex && !isNumeric(row.get(i))) {
          categoricalColumns.add(i);
        }
      }
      String label = row.get(labelIndex).trim();
      if (label.isBlank()) {
        blankLabelRows += 1;
      } else {
        labelCounts.merge(label, 1, Integer::sum);
      }
    }

    if (labelCounts.isEmpty()) {
      throw new IllegalArgumentException("CSV label column cannot be empty.");
    }
    List<String> labels = new ArrayList<>(labelCounts.keySet());
    if (labels.size() < MIN_CLASS_COUNT) {
      throw new IllegalArgumentException("CSV classification dataset requires at least two classes.");
    }
    if (labels.size() > classCount) {
      throw new IllegalArgumentException("CSV label column has " + labels.size() + " distinct labels, which exceeds the configured class count " + classCount + ".");
    }
    int usableRows = rows.size() - blankLabelRows;
    if (usableRows < MIN_ROWS) {
      throw new IllegalArgumentException("CSV dataset requires at least " + MIN_ROWS + " labeled data rows.");
    }
    try {
      Files.createDirectories(datasetDir);
      Files.write(csvTarget, bytes);
      Files.writeString(datasetDir.resolve("label-column.txt"), headers.get(labelIndex), StandardCharsets.UTF_8);
      Files.writeString(datasetDir.resolve("class-count.txt"), String.valueOf(classCount), StandardCharsets.UTF_8);
    } catch (IOException ex) {
      throw new IllegalArgumentException("Failed to save CSV dataset.");
    }

    List<String> warnings = new ArrayList<>();
    if (missingValues > 0) {
      warnings.add("发现 " + missingValues + " 个缺失值，训练前建议清洗或填补。");
    }
    if (blankLabelRows > 0) {
      warnings.add("标签列中有 " + blankLabelRows + " 行为空，训练时会自动跳过这些样本。");
    }
    if (!categoricalColumns.isEmpty()) {
      warnings.add("检测到 " + categoricalColumns.size() + " 个非数值特征列，训练时会进行简单类别编码。");
    }
    if (classCount > labels.size()) {
      warnings.add("用户设置类别数为 " + classCount + "，当前 CSV 中实际出现 " + labels.size() + " 个标签值。");
    }
    warnings.addAll(imbalanceWarnings(labelCounts));

    int featureColumns = encodedCsvFeatureCount(headers, rows, labelIndex);
    if (featureColumns <= 0) {
      throw new IllegalArgumentException("CSV must contain at least one usable feature column.");
    }
    return new TrainingDatasetDetail(
        datasetId,
        file.getOriginalFilename() == null ? "uploaded.csv" : file.getOriginalFilename(),
        "upload",
        "table",
        "用户上传 CSV 数据集",
        usableRows,
        classCount,
        featureColumns + " encoded features",
        "70% / 15% / 15%",
        labels,
        true,
        0.7,
        0.15,
        0.15,
        mapToDistribution(labelCounts),
        null,
        new TablePreview(headers, rows.stream().limit(6).toList()),
        null,
        warnings
    );
  }

  private TrainingDatasetDetail importImages(List<MultipartFile> files) {
    if (files.size() > MAX_IMAGE_COUNT) {
      throw new IllegalArgumentException("Image dataset has too many files. Please keep it under " + MAX_IMAGE_COUNT + " images.");
    }
    String datasetId = nextUploadId();
    Path imageRoot = uploadDatasetDir(datasetId).resolve("images").normalize();
    ensureUnder(uploadDatasetDir(datasetId), imageRoot);

    Map<String, Integer> labelCounts = new LinkedHashMap<>();
    Map<String, List<ImagePreviewItem>> previewsByLabel = new LinkedHashMap<>();
    Set<String> sizes = new java.util.LinkedHashSet<>();
    try {
      Files.createDirectories(imageRoot);
      for (int i = 0; i < files.size(); i += 1) {
        MultipartFile file = files.get(i);
        String originalName = file.getOriginalFilename() == null ? "image-" + i + ".png" : Path.of(file.getOriginalFilename()).getFileName().toString();
        String label = labelFromImageName(file.getOriginalFilename());
        if ("未标注".equals(label)) {
          throw new IllegalArgumentException("Image filename must start with a class label, for example cat_001.jpg.");
        }
        labelCounts.merge(label, 1, Integer::sum);
        ImageInfo imageInfo = readImageInfo(file);
        if (imageInfo == null) {
          throw new IllegalArgumentException("Invalid image file: " + originalName);
        }
        sizes.add(imageInfo.width() + "x" + imageInfo.height());
        Path labelDir = imageRoot.resolve(sanitizePathSegment(label)).normalize();
        ensureUnder(imageRoot, labelDir);
        Files.createDirectories(labelDir);
        String filename = sanitizeFilename(i + "-" + originalName);
        Path target = labelDir.resolve(filename).normalize();
        ensureUnder(labelDir, target);
        try (InputStream input = file.getInputStream()) {
          Files.copy(input, target, StandardCopyOption.REPLACE_EXISTING);
        }
        addImagePreview(
            previewsByLabel,
            label,
            new ImagePreviewItem(originalName, label, "/datasets/upload/" + datasetId + "/images/" + sanitizePathSegment(label) + "/" + filename)
        );
      }
    } catch (IOException ex) {
      throw new IllegalArgumentException("Failed to save uploaded image previews.");
    }

    validateImageLabels(labelCounts);

    List<String> labels = labelCounts.keySet().stream().filter(label -> !"未标注".equals(label)).toList();
    boolean hasLabels = true;
    List<String> warnings = new ArrayList<>();
    if (sizes.size() > 1) {
      warnings.add("检测到图片尺寸不一致，训练前需要统一 resize。");
    }
    warnings.addAll(imbalanceWarnings(labelCounts));

    String inputShape = sizes.size() == 1 ? sizes.iterator().next() + " x 3" : "mixed image sizes";
    return new TrainingDatasetDetail(
        datasetId,
        "图片导入 " + files.size() + " 张",
        "upload",
        "image",
        "用户上传图片数据集",
        files.size(),
        labels.size(),
        inputShape,
        "70% / 15% / 15%",
        labels,
        hasLabels,
        0.7,
        0.15,
        0.15,
        mapToDistribution(labelCounts),
        flattenPreviewGroups(previewsByLabel),
        null,
        null,
        warnings
    );
  }

  private TrainingDatasetDetail importImageZip(MultipartFile file) {
    String datasetId = nextUploadId();
    Path imageRoot = uploadDatasetDir(datasetId).resolve("images").normalize();
    ensureUnder(uploadDatasetDir(datasetId), imageRoot);

    Map<String, Integer> labelCounts = new LinkedHashMap<>();
    Map<String, List<ImagePreviewItem>> previewsByLabel = new LinkedHashMap<>();
    Set<String> sizes = new java.util.LinkedHashSet<>();
    int imageCount = 0;
    try {
      Files.createDirectories(imageRoot);
      try (ZipInputStream zip = new ZipInputStream(file.getInputStream(), StandardCharsets.UTF_8)) {
        ZipEntry entry;
        while ((entry = zip.getNextEntry()) != null) {
          if (entry.isDirectory()) {
            continue;
          }
          String entryName = entry.getName().replace('\\', '/');
          if (entryName.startsWith("/") || entryName.contains("../")) {
            throw new IllegalArgumentException("ZIP contains unsafe paths.");
          }
          if (isIgnoredZipEntry(entryName)) {
            continue;
          }
          String[] parts = entryName.split("/");
          String filename = parts.length == 0 ? "" : parts[parts.length - 1].trim();
          if (filename.isBlank() || !isLocalPreviewImage(filename)) {
            continue;
          }
          if (parts.length < 2) {
            throw new IllegalArgumentException("ZIP image datasets must use class folders, for example cat/001.jpg.");
          }
          String label = parts[parts.length - 2].trim();
          if (label.isBlank()) {
            continue;
          }
          imageCount += 1;
          if (imageCount > MAX_IMAGE_COUNT) {
            throw new IllegalArgumentException("Image dataset has too many files. Please keep it under " + MAX_IMAGE_COUNT + " images.");
          }
          byte[] bytes = zip.readAllBytes();
          ImageInfo imageInfo = readImageInfo(bytes);
          if (imageInfo == null) {
            throw new IllegalArgumentException("Invalid image in ZIP: " + entryName);
          }
          sizes.add(imageInfo.width() + "x" + imageInfo.height());
          labelCounts.merge(label, 1, Integer::sum);

          Path labelDir = imageRoot.resolve(sanitizePathSegment(label)).normalize();
          ensureUnder(imageRoot, labelDir);
          Files.createDirectories(labelDir);
          String storedFilename = sanitizeFilename(imageCount + "-" + filename);
          Path target = labelDir.resolve(storedFilename).normalize();
          ensureUnder(labelDir, target);
          Files.write(target, bytes);
          addImagePreview(
              previewsByLabel,
              label,
              new ImagePreviewItem(filename, label, "/datasets/upload/" + datasetId + "/images/" + sanitizePathSegment(label) + "/" + storedFilename)
          );
        }
      }
    } catch (IOException ex) {
      throw new IllegalArgumentException("Failed to read ZIP dataset.");
    }

    if (imageCount == 0) {
      throw new IllegalArgumentException("ZIP contains no supported images.");
    }
    validateImageLabels(labelCounts);

    List<String> labels = new ArrayList<>(labelCounts.keySet());
    List<String> warnings = new ArrayList<>();
    if (sizes.size() > 1) {
      warnings.add("检测到图片尺寸不一致，训练前会按输入层尺寸统一 resize。");
    }
    warnings.addAll(imbalanceWarnings(labelCounts));
    String inputShape = sizes.size() == 1 ? sizes.iterator().next() + " x 3" : "mixed image sizes";
    return new TrainingDatasetDetail(
        datasetId,
        file.getOriginalFilename() == null ? "ZIP 图片数据集" : file.getOriginalFilename(),
        "upload",
        "image",
        "用户上传 ZIP 图片分类数据集",
        imageCount,
        labels.size(),
        inputShape,
        "70% / 15% / 15%",
        labels,
        true,
        0.7,
        0.15,
        0.15,
        mapToDistribution(labelCounts),
        flattenPreviewGroups(previewsByLabel),
        null,
        null,
        warnings
    );
  }

  private void registerBuiltinDatasets() {
    saveBuiltinIfMissing(builtinImage(
        "mnist-1000",
        "MNIST 全量",
        "28x28 灰度手写数字，全量训练集与测试集。",
        70000,
        "28 x 28 x 1",
        List.of("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    ));
    saveBuiltinIfMissing(builtinImage(
        "cifar10-500",
        "CIFAR-10 全量",
        "32x32 RGB 彩色图片，全量训练集与测试集，覆盖 10 个常见物体类别。",
        60000,
        "32 x 32 x 3",
        List.of("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    ));
    saveBuiltinIfMissing(builtinImage(
        "cifar10-5000",
        "CIFAR-10 5000 张",
        "从 CIFAR-10 全量数据中按类别均衡抽取 5000 张图片，适合课堂快速训练演示。",
        5000,
        "32 x 32 x 3",
        List.of("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    ));

    List<String> irisLabels = List.of("setosa", "versicolor", "virginica");
    saveBuiltinIfMissing(new TrainingDatasetDetail(
        "iris",
        "鸢尾花数据集",
        "builtin",
        "table",
        "4 维表格特征，适合全连接网络分类演示。",
        150,
        3,
        "4 numeric features",
        "80% / 20%",
        irisLabels,
        true,
        0.8,
        0,
        0.2,
        evenDistribution(irisLabels, 150),
        null,
        irisTablePreview(),
        null,
        List.of()
    ));

    List<String> pointLabels = List.of("class A", "class B");
    saveBuiltinIfMissing(new TrainingDatasetDetail(
        "points-2d",
        "二维分类数据集",
        "builtin",
        "points",
        "二维坐标点，适合展示决策边界和二分类过程。",
        300,
        2,
        "x, y",
        "70% / 15% / 15%",
        pointLabels,
        true,
        0.7,
        0.15,
        0.15,
        evenDistribution(pointLabels, 300),
        null,
        null,
        makePointPreview(),
        List.of()
    ));
  }

  private void saveBuiltinIfMissing(TrainingDatasetDetail detail) {
    datasets.save(toEntity(detail));
  }

  private TrainingDataset toEntity(TrainingDatasetDetail detail) {
    return new TrainingDataset(
        detail.id(),
        detail.name(),
        detail.source(),
        detail.kind(),
        detail.description(),
        detail.sampleCount(),
        detail.classCount(),
        detail.inputShape(),
        detail.recommendedSplit(),
        writeJson(detail.labels()),
        detail.hasLabels(),
        detail.trainRatio(),
        detail.valRatio(),
        detail.testRatio(),
        writeJson(detail.labelDistribution()),
        writeJson(detail.imagePreview()),
        writeJson(detail.tablePreview()),
        writeJson(detail.pointPreview()),
        writeJson(detail.warnings())
    );
  }

  private TrainingDatasetDetail toDetail(TrainingDataset row) {
    int classCount = row.getClassCount();
    if ("upload".equals(row.getSource()) && "table".equals(row.getKind())) {
      classCount = readUploadClassCount(row.getId()).orElse(classCount);
    }
    return new TrainingDatasetDetail(
        row.getId(),
        row.getName(),
        row.getSource(),
        row.getKind(),
        row.getDescription(),
        row.getSampleCount(),
        classCount,
        row.getInputShape(),
        row.getRecommendedSplit(),
        readList(row.getLabelsJson(), String.class),
        row.isHasLabels(),
        row.getTrainRatio(),
        row.getValRatio(),
        row.getTestRatio(),
        readList(row.getLabelDistributionJson(), LabelDistributionItem.class),
        readList(row.getImagePreviewJson(), ImagePreviewItem.class),
        readNullable(row.getTablePreviewJson(), TablePreview.class),
        readList(row.getPointPreviewJson(), PointPreviewItem.class),
        readList(row.getWarningsJson(), String.class)
    );
  }

  private java.util.Optional<Integer> readUploadClassCount(String datasetId) {
    Path path = uploadDatasetDir(datasetId).resolve("class-count.txt").normalize();
    ensureUnder(uploadDatasetDir(datasetId), path);
    if (!Files.isRegularFile(path)) {
      return java.util.Optional.empty();
    }
    try {
      int value = Integer.parseInt(Files.readString(path, StandardCharsets.UTF_8).trim());
      return value >= MIN_CLASS_COUNT ? java.util.Optional.of(value) : java.util.Optional.empty();
    } catch (IOException | NumberFormatException ex) {
      return java.util.Optional.empty();
    }
  }

  private String writeJson(Object value) {
    try {
      return objectMapper.writeValueAsString(value == null ? List.of() : value);
    } catch (JsonProcessingException ex) {
      throw new IllegalArgumentException("Failed to serialize dataset metadata.");
    }
  }

  private <T> List<T> readList(String json, Class<T> itemType) {
    if (json == null || json.isBlank() || "null".equals(json)) {
      return List.of();
    }
    try {
      return objectMapper.readValue(
          json,
          objectMapper.getTypeFactory().constructCollectionType(List.class, itemType)
      );
    } catch (IOException ex) {
      return List.of();
    }
  }

  private <T> T readNullable(String json, Class<T> itemType) {
    if (json == null || json.isBlank() || "null".equals(json) || "[]".equals(json)) {
      return null;
    }
    try {
      return objectMapper.readValue(json, itemType);
    } catch (IOException ex) {
      return null;
    }
  }

  private TrainingDatasetDetail builtinImage(
      String id,
      String name,
      String description,
      int sampleCount,
      String inputShape,
      List<String> labels
  ) {
    List<ImagePreviewItem> previews = new ArrayList<>();
    List<ImagePreviewItem> localPreviews = localImagePreviews(id);
    if (!localPreviews.isEmpty()) {
      previews.addAll(localPreviews);
      return new TrainingDatasetDetail(
          id, name, "builtin", "image", description, sampleCount, labels.size(), inputShape,
          "70% / 15% / 15%", labels, true, 0.7, 0.15, 0.15,
          evenDistribution(labels, sampleCount), previews, null, null, List.of()
      );
    }
    for (int i = 0; i < labels.size(); i += 1) {
      String label = labels.get(i);
      for (int sample = 0; sample < PREVIEW_IMAGES_PER_CLASS; sample += 1) {
        int previewIndex = i * PREVIEW_IMAGES_PER_CLASS + sample + 1;
        previews.add(new ImagePreviewItem(label + "_" + sample + ".png", label, "/api/training/datasets/" + id + "/preview/" + previewIndex));
      }
    }
    return new TrainingDatasetDetail(
        id, name, "builtin", "image", description, sampleCount, labels.size(), inputShape,
        "70% / 15% / 15%", labels, true, 0.7, 0.15, 0.15,
        evenDistribution(labels, sampleCount), previews, null, null, List.of()
    );
  }

  private List<ImagePreviewItem> localImagePreviews(String datasetId) {
    Path imagesRoot = datasetsRoot.resolve("builtin").resolve(datasetId).resolve("images").normalize();
    if (!imagesRoot.startsWith(datasetsRoot) || !Files.isDirectory(imagesRoot)) {
      return List.of();
    }
    List<ImagePreviewItem> previews = new ArrayList<>();
    try (var labels = Files.list(imagesRoot)) {
      List<Path> labelDirs = labels
          .filter(Files::isDirectory)
          .sorted()
          .toList();
      for (Path labelDir : labelDirs) {
        String label = labelDir.getFileName().toString();
        try (var files = Files.list(labelDir)) {
          List<Path> images = files
              .filter(Files::isRegularFile)
              .filter(path -> isLocalPreviewImage(path.getFileName().toString()))
              .sorted()
              .limit(PREVIEW_IMAGES_PER_CLASS)
              .toList();
          for (Path image : images) {
            String filename = image.getFileName().toString();
            previews.add(new ImagePreviewItem(
                filename,
                label,
                "/datasets/builtin/" + datasetId + "/images/" + label + "/" + filename
            ));
          }
        }
      }
    } catch (IOException ex) {
      return List.of();
    }
    return previews;
  }

  private void addImagePreview(Map<String, List<ImagePreviewItem>> previewsByLabel, String label, ImagePreviewItem item) {
    List<ImagePreviewItem> items = previewsByLabel.computeIfAbsent(label, ignored -> new ArrayList<>());
    if (items.size() < PREVIEW_IMAGES_PER_CLASS) {
      items.add(item);
    }
  }

  private List<ImagePreviewItem> flattenPreviewGroups(Map<String, List<ImagePreviewItem>> previewsByLabel) {
    return previewsByLabel.values().stream()
        .flatMap(List::stream)
        .toList();
  }

  private boolean isLocalPreviewImage(String filename) {
    int dot = filename.lastIndexOf('.');
    if (dot < 0) {
      return false;
    }
    return IMAGE_EXTENSIONS.contains(filename.substring(dot + 1).toLowerCase(Locale.ROOT));
  }

  private boolean isIgnoredZipEntry(String entryName) {
    String normalized = entryName.replace('\\', '/');
    if (normalized.startsWith("__MACOSX/") || normalized.contains("/__MACOSX/")) {
      return true;
    }
    for (String part : normalized.split("/")) {
      if (part.equals(".DS_Store") || part.startsWith("._") || part.equals("Thumbs.db")) {
        return true;
      }
    }
    return false;
  }

  private List<PointPreviewItem> makePointPreview() {
    List<PointPreviewItem> points = new ArrayList<>();
    for (int i = 0; i < 36; i += 1) {
      double angle = i * 0.45;
      double radius = 0.18 + (i % 9) * 0.035;
      String label = i % 2 == 0 ? "class A" : "class B";
      double x = Math.cos(angle) * radius + (i % 2 == 0 ? -0.22 : 0.22);
      double y = Math.sin(angle) * radius + (i % 2 == 0 ? -0.12 : 0.12);
      points.add(new PointPreviewItem(round(x, 3), round(y, 3), label, i % 2 == 0 ? COLORS.get(0) : COLORS.get(1)));
    }
    return points;
  }

  private TablePreview irisTablePreview() {
    Path path = datasetsRoot.resolve("builtin").resolve("iris").resolve("iris.csv").normalize();
    try {
      List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8).stream()
          .map(String::trim)
          .filter(line -> !line.isBlank())
          .toList();
      if (lines.size() >= 2) {
        List<String> headers = parseCsvLine(lines.get(0));
        List<List<String>> rows = lines.stream()
            .skip(1)
            .limit(8)
            .map(this::parseCsvLine)
            .filter(row -> !row.isEmpty())
            .toList();
        return new TablePreview(headers, rows);
      }
    } catch (IOException ignored) {
      // Fall through to the bundled static preview.
    }
    return new TablePreview(
        List.of("sepal_length", "sepal_width", "petal_length", "petal_width", "label"),
        List.of(
            List.of("5.1", "3.5", "1.4", "0.2", "setosa"),
            List.of("6.4", "3.2", "4.5", "1.5", "versicolor"),
            List.of("6.3", "3.3", "6.0", "2.5", "virginica"),
            List.of("5.8", "2.7", "4.1", "1.0", "versicolor")
        )
    );
  }

  private List<LabelDistributionItem> evenDistribution(List<String> labels, int sampleCount) {
    Map<String, Integer> counts = new LinkedHashMap<>();
    int base = sampleCount / Math.max(1, labels.size());
    int extra = sampleCount % Math.max(1, labels.size());
    for (int i = 0; i < labels.size(); i += 1) {
      counts.put(labels.get(i), base + (i < extra ? 1 : 0));
    }
    return mapToDistribution(counts);
  }

  private List<LabelDistributionItem> mapToDistribution(Map<String, Integer> counts) {
    List<LabelDistributionItem> result = new ArrayList<>();
    int i = 0;
    for (Map.Entry<String, Integer> entry : counts.entrySet()) {
      result.add(new LabelDistributionItem(entry.getKey(), entry.getValue(), COLORS.get(i % COLORS.size())));
      i += 1;
    }
    return result;
  }

  private List<String> imbalanceWarnings(Map<String, Integer> counts) {
    if (counts.size() < 2) {
      return List.of();
    }
    int min = counts.values().stream().min(Comparator.naturalOrder()).orElse(0);
    int max = counts.values().stream().max(Comparator.naturalOrder()).orElse(0);
    if (min > 0 && max >= min * 3) {
      return List.of("类别分布不均衡，最大类别样本数至少是最小类别的 3 倍。");
    }
    return List.of();
  }

  private int encodedCsvFeatureCount(List<String> headers, List<List<String>> rows, int labelIndex) {
    int count = 0;
    for (int i = 0; i < headers.size(); i += 1) {
      if (i == labelIndex || isIgnoredCsvFeatureColumn(headers.get(i))) {
        continue;
      }
      Set<String> values = new java.util.LinkedHashSet<>();
      boolean numeric = true;
      for (List<String> row : rows) {
        String value = i < row.size() ? row.get(i).trim() : "";
        if (value.isBlank()) {
          continue;
        }
        values.add(value);
        if (!isNumeric(value)) {
          numeric = false;
        }
      }
      if (values.isEmpty()) {
        continue;
      }
      count += numeric ? 1 : values.size();
    }
    return count;
  }

  private boolean isIgnoredCsvFeatureColumn(String header) {
    String normalized = header == null ? "" : header.trim().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9\\p{IsHan}]", "");
    return normalized.equals("id")
        || normalized.equals("studentid")
        || normalized.equals("name")
        || normalized.equals("姓名")
        || normalized.endsWith("id");
  }

  private boolean isCsvFile(MultipartFile file) {
    String name = safeLowerName(file);
    String type = file.getContentType() == null ? "" : file.getContentType().toLowerCase(Locale.ROOT);
    return name.endsWith(".csv") || "text/csv".equals(type) || "application/vnd.ms-excel".equals(type);
  }

  private boolean isZipFile(MultipartFile file) {
    String name = safeLowerName(file);
    String type = file.getContentType() == null ? "" : file.getContentType().toLowerCase(Locale.ROOT);
    return name.endsWith(".zip")
        || "application/zip".equals(type)
        || "application/x-zip-compressed".equals(type);
  }

  private boolean isImageFile(MultipartFile file) {
    String name = safeLowerName(file);
    String type = file.getContentType() == null ? "" : file.getContentType().toLowerCase(Locale.ROOT);
    int dot = name.lastIndexOf('.');
    String extension = dot >= 0 ? name.substring(dot + 1) : "";
    return type.startsWith("image/") || IMAGE_EXTENSIONS.contains(extension);
  }

  private boolean isNumeric(String value) {
    try {
      Double.parseDouble(value.trim());
      return true;
    } catch (NumberFormatException ex) {
      return false;
    }
  }

  private void validateImageLabels(Map<String, Integer> labelCounts) {
    List<String> labels = labelCounts.keySet().stream().filter(label -> !"未标注".equals(label)).toList();
    if (labels.size() < MIN_CLASS_COUNT) {
      throw new IllegalArgumentException("Image classification dataset requires at least two class labels.");
    }
    for (String label : labels) {
      int count = labelCounts.getOrDefault(label, 0);
      if (count < MIN_SAMPLES_PER_IMAGE_CLASS) {
        throw new IllegalArgumentException("Each image class needs at least " + MIN_SAMPLES_PER_IMAGE_CLASS + " images. Class " + label + " has " + count + ".");
      }
    }
  }

  private Path uploadDatasetDir(String datasetId) {
    Path dir = datasetsRoot.resolve("upload").resolve(datasetId).normalize();
    ensureUnder(datasetsRoot, dir);
    return dir;
  }

  private void ensureUnder(Path root, Path target) {
    if (!target.normalize().startsWith(root.normalize())) {
      throw new IllegalArgumentException("Invalid dataset path.");
    }
  }

  private void deleteDirectoryIfExists(Path directory) throws IOException {
    Path uploadRoot = datasetsRoot.resolve("upload").normalize();
    ensureUnder(uploadRoot, directory);
    if (!Files.exists(directory)) {
      return;
    }
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }

  private String safeLowerName(MultipartFile file) {
    String name = file.getOriginalFilename() == null ? "" : file.getOriginalFilename();
    return name.toLowerCase(Locale.ROOT);
  }

  private List<String> parseCsvLine(String line) {
    List<String> cells = new ArrayList<>();
    StringBuilder current = new StringBuilder();
    boolean quoted = false;
    for (int i = 0; i < line.length(); i += 1) {
      char ch = line.charAt(i);
      if (ch == '"') {
        if (quoted && i + 1 < line.length() && line.charAt(i + 1) == '"') {
          current.append('"');
          i += 1;
        } else {
          quoted = !quoted;
        }
      } else if (ch == ',' && !quoted) {
        cells.add(current.toString().trim());
        current.setLength(0);
      } else {
        current.append(ch);
      }
    }
    cells.add(current.toString().trim());
    return cells;
  }

  private int resolveRequiredLabelColumn(List<String> headers, String labelColumn) {
    if (labelColumn == null || labelColumn.isBlank()) {
      throw new IllegalArgumentException("Please choose a CSV label column before importing.");
    }
    String requested = labelColumn.trim();
    for (int i = 0; i < headers.size(); i += 1) {
      if (headers.get(i).trim().equals(requested)) {
        return i;
      }
    }
    String normalizedRequested = normalizeColumnName(requested);
    for (int i = 0; i < headers.size(); i += 1) {
      if (normalizeColumnName(headers.get(i)).equals(normalizedRequested)) {
        return i;
      }
    }
    throw new IllegalArgumentException("Selected CSV label column does not exist: " + requested);
  }

  private int resolveRequiredClassCount(Integer classCount) {
    if (classCount == null) {
      throw new IllegalArgumentException("Please enter CSV class count before importing.");
    }
    if (classCount < MIN_CLASS_COUNT) {
      throw new IllegalArgumentException("CSV class count must be at least " + MIN_CLASS_COUNT + ".");
    }
    return classCount;
  }

  private String normalizeColumnName(String value) {
    return value == null ? "" : value.trim().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
  }

  private String labelFromImageName(String originalName) {
    if (originalName == null || originalName.isBlank()) {
      return "未标注";
    }
    String normalized = originalName.replace('\\', '/');
    int slash = normalized.lastIndexOf('/');
    if (slash > 0) {
      String parentPart = normalized.substring(0, slash);
      int parentSlash = parentPart.lastIndexOf('/');
      String parent = parentSlash >= 0 ? parentPart.substring(parentSlash + 1) : parentPart;
      if (!parent.isBlank()) {
        return parent;
      }
    }
    String filename = slash >= 0 ? normalized.substring(slash + 1) : normalized;
    int dot = filename.lastIndexOf('.');
    String stem = dot > 0 ? filename.substring(0, dot) : filename;
    int split = firstPositive(stem.indexOf('_'), stem.indexOf('-'), stem.indexOf(' '));
    return split > 0 ? stem.substring(0, split) : "未标注";
  }

  private int firstPositive(int... values) {
    int result = -1;
    for (int value : values) {
      if (value > 0 && (result < 0 || value < result)) {
        result = value;
      }
    }
    return result;
  }

  private ImageInfo readImageInfo(MultipartFile file) {
    try (InputStream input = file.getInputStream()) {
      BufferedImage image = ImageIO.read(input);
      return image == null ? null : new ImageInfo(image.getWidth(), image.getHeight());
    } catch (IOException ex) {
      return null;
    }
  }

  private ImageInfo readImageInfo(byte[] bytes) {
    try (InputStream input = new ByteArrayInputStream(bytes)) {
      BufferedImage image = ImageIO.read(input);
      return image == null ? null : new ImageInfo(image.getWidth(), image.getHeight());
    } catch (IOException ex) {
      return null;
    }
  }

  private String nextUploadId() {
    return "upload-" + LocalDateTime.now().format(UPLOAD_ID_TIME) + "-" + UUID.randomUUID().toString().substring(0, 8);
  }

  private String sanitizeFilename(String filename) {
    String encoded = URLEncoder.encode(filename, StandardCharsets.UTF_8).replace("+", "%20");
    return encoded.replace("%", "_");
  }

  private String sanitizePathSegment(String value) {
    String cleaned = value == null || value.isBlank() ? "unknown" : value.trim();
    cleaned = cleaned.replace('\\', '_').replace('/', '_').replace("..", "_");
    return sanitizeFilename(cleaned);
  }

  private double round(double value, int digits) {
    double scale = Math.pow(10, digits);
    return Math.round(value * scale) / scale;
  }

  private static String escapeXml(String value) {
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
  }

  private record ImageInfo(int width, int height) {}
}
