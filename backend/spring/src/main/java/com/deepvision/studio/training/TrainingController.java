package com.deepvision.studio.training;

import com.deepvision.studio.training.TrainingDtos.CheckpointTestResult;
import com.deepvision.studio.training.TrainingDtos.CollaborationRoomSummary;
import com.deepvision.studio.training.TrainingDtos.DatasetErrorResponse;
import com.deepvision.studio.training.TrainingDtos.DatasetImportResponse;
import com.deepvision.studio.training.TrainingDtos.InferenceSampleListResponse;
import com.deepvision.studio.training.TrainingDtos.SingleInferenceRequest;
import com.deepvision.studio.training.TrainingDtos.SingleInferenceResult;
import com.deepvision.studio.training.TrainingDtos.StartTrainingRequest;
import com.deepvision.studio.training.TrainingDtos.TestCheckpointRequest;
import com.deepvision.studio.training.TrainingDtos.TrainingCheckpointSummary;
import com.deepvision.studio.training.TrainingDtos.TrainingControlResponse;
import com.deepvision.studio.training.TrainingDtos.TrainingDatasetDetail;
import com.deepvision.studio.training.TrainingDtos.TrainingDatasetOption;
import com.deepvision.studio.training.TrainingDtos.TrainingStartResponse;
import com.deepvision.studio.training.TrainingDtos.TrainingStatusResponse;
import com.deepvision.studio.training.TrainingDtos.WeightHistogramResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.Principal;
import java.util.List;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/training")
@Tag(name = "Training", description = "Dataset, training job, checkpoint, and collaboration REST APIs")
public class TrainingController {
  private final TrainingDatasetService datasetService;
  private final TrainingJobService jobService;
  private final TrainingCollaborationHandler collaborationHandler;

  public TrainingController(
      TrainingDatasetService datasetService,
      TrainingJobService jobService,
      TrainingCollaborationHandler collaborationHandler
  ) {
    this.datasetService = datasetService;
    this.jobService = jobService;
    this.collaborationHandler = collaborationHandler;
  }

  @GetMapping("/datasets")
  @Operation(summary = "List training datasets")
  public List<TrainingDatasetOption> listDatasets(
      Principal principal,
      @Parameter(description = "Optional source filter, such as builtin or upload")
      @RequestParam(value = "source", required = false) String source
  ) {
    return datasetService.listDatasets(source, username(principal));
  }

  @GetMapping("/datasets/builtin")
  @Operation(summary = "List built-in training datasets")
  public List<TrainingDatasetOption> listBuiltinDatasets() {
    return datasetService.listBuiltin();
  }

  @GetMapping("/datasets/{datasetId}")
  @Operation(summary = "Get dataset detail")
  @ApiResponse(responseCode = "200", description = "Dataset detail")
  @ApiResponse(responseCode = "400", description = "Dataset not found")
  public TrainingDatasetDetail datasetDetail(Principal principal, @PathVariable String datasetId) {
    return datasetService.getDetail(datasetId, username(principal));
  }

  @DeleteMapping("/datasets/{datasetId}")
  @Operation(summary = "Delete an uploaded dataset")
  @ApiResponse(responseCode = "204", description = "Dataset deleted")
  @ApiResponse(responseCode = "400", description = "Dataset cannot be deleted or does not exist")
  public ResponseEntity<Void> deleteDataset(Principal principal, @PathVariable String datasetId) {
    datasetService.deleteUploadedDataset(datasetId, username(principal));
    return ResponseEntity.noContent().build();
  }

  @GetMapping("/datasets/{datasetId}/files/**")
  @Operation(summary = "Get a private uploaded dataset file")
  @ApiResponse(responseCode = "200", description = "Dataset file")
  @ApiResponse(responseCode = "400", description = "Dataset file is invalid or not visible")
  public ResponseEntity<Resource> uploadedDatasetFile(
      Principal principal,
      @PathVariable String datasetId,
      jakarta.servlet.http.HttpServletRequest request
  ) throws IOException {
    String prefix = "/api/training/datasets/" + datasetId + "/files/";
    String uri = request.getRequestURI();
    int offset = uri.indexOf(prefix);
    String relativePath = offset < 0 ? "" : uri.substring(offset + prefix.length());
    Path file = datasetService.uploadDatasetFile(datasetId, java.net.URLDecoder.decode(relativePath, java.nio.charset.StandardCharsets.UTF_8), username(principal));
    String contentType = Files.probeContentType(file);
    MediaType mediaType = contentType == null ? MediaType.APPLICATION_OCTET_STREAM : MediaType.parseMediaType(contentType);
    return ResponseEntity.ok()
        .contentType(mediaType)
        .header(HttpHeaders.CACHE_CONTROL, "private, max-age=300")
        .body(new FileSystemResource(file));
  }

  @GetMapping(value = "/datasets/{datasetId}/preview/{index}", produces = "image/svg+xml")
  @Operation(summary = "Get SVG preview for a built-in dataset sample")
  @ApiResponse(responseCode = "200", description = "SVG preview")
  @ApiResponse(responseCode = "400", description = "Dataset or preview index is invalid")
  public ResponseEntity<String> builtInPreview(@PathVariable String datasetId, @PathVariable int index) {
    return ResponseEntity.ok()
        .header(HttpHeaders.CACHE_CONTROL, "public, max-age=3600")
        .body(datasetService.builtInPreviewSvg(datasetId, index));
  }

  @PostMapping(value = "/datasets/imports", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
  @Operation(summary = "Import an uploaded training dataset")
  @ApiResponse(responseCode = "200", description = "Dataset imported")
  @ApiResponse(responseCode = "400", description = "Files or import options are invalid")
  @ApiResponse(responseCode = "413", description = "Uploaded files exceed configured size limits")
  public DatasetImportResponse importDataset(
      Principal principal,
      @RequestParam("files") MultipartFile[] files,
      @RequestParam(value = "labelColumn", required = false) String labelColumn,
      @RequestParam(value = "classCount", required = false) Integer classCount
  ) {
    return datasetService.importDataset(files, labelColumn, classCount, username(principal));
  }

  @PostMapping("/start")
  @Operation(summary = "Start a training job")
  @ApiResponse(responseCode = "200", description = "Training job started")
  @ApiResponse(responseCode = "400", description = "Training configuration is invalid")
  public TrainingStartResponse start(Principal principal, @Valid @RequestBody StartTrainingRequest request) {
    return jobService.start(request, principal == null ? null : principal.getName());
  }

  @GetMapping("/checkpoints")
  @Operation(summary = "List saved training checkpoints")
  public List<TrainingCheckpointSummary> checkpoints(
      Principal principal,
      @RequestParam(value = "datasetId", required = false) String datasetId
  ) {
    return jobService.listCheckpoints(principal == null ? null : principal.getName(), datasetId);
  }

  @GetMapping("/collaboration/rooms")
  @Operation(summary = "List active training collaboration rooms")
  public List<CollaborationRoomSummary> collaborationRooms() {
    return collaborationHandler.listRooms();
  }

  @PostMapping("/checkpoints/{checkpointId}/test")
  @Operation(summary = "Run checkpoint test evaluation")
  @ApiResponse(responseCode = "200", description = "Checkpoint test result")
  @ApiResponse(responseCode = "400", description = "Checkpoint or dataset is invalid")
  public CheckpointTestResult testCheckpoint(
      Principal principal,
      @PathVariable Long checkpointId,
      @Valid @RequestBody TestCheckpointRequest request
  ) {
    return jobService.testCheckpoint(principal == null ? null : principal.getName(), checkpointId, request);
  }

  @GetMapping("/checkpoints/{checkpointId}/samples")
  @Operation(summary = "List samples available for single checkpoint inference")
  @ApiResponse(responseCode = "200", description = "Sample list")
  @ApiResponse(responseCode = "400", description = "Checkpoint is invalid")
  public InferenceSampleListResponse checkpointSamples(
      Principal principal,
      @PathVariable Long checkpointId,
      @RequestParam(value = "limit", required = false, defaultValue = "60") int limit
  ) {
    return jobService.listCheckpointSamples(principal == null ? null : principal.getName(), checkpointId, limit);
  }

  @PostMapping("/checkpoints/{checkpointId}/infer")
  @Operation(summary = "Run single-sample inference with layer activations")
  @ApiResponse(responseCode = "200", description = "Single inference result")
  @ApiResponse(responseCode = "400", description = "Checkpoint or sample is invalid")
  public SingleInferenceResult inferCheckpointSample(
      Principal principal,
      @PathVariable Long checkpointId,
      @Valid @RequestBody SingleInferenceRequest request
  ) {
    return jobService.inferCheckpointSample(principal == null ? null : principal.getName(), checkpointId, request);
  }

  @GetMapping("/{jobId}/weights/histogram")
  @Operation(summary = "Get weight histogram for a training job")
  public WeightHistogramResponse histogram(Principal principal, @PathVariable String jobId) {
    return jobService.histogram(username(principal), jobId);
  }

  @GetMapping("/{jobId}/status")
  @Operation(summary = "Get training job status")
  public TrainingStatusResponse status(Principal principal, @PathVariable String jobId) {
    return jobService.status(username(principal), jobId);
  }

  @PostMapping("/{jobId}/pause")
  @Operation(summary = "Pause a training job")
  public TrainingControlResponse pause(Principal principal, @PathVariable String jobId) {
    return jobService.pause(username(principal), jobId);
  }

  @PostMapping("/{jobId}/resume")
  @Operation(summary = "Resume a paused training job")
  public TrainingControlResponse resume(Principal principal, @PathVariable String jobId) {
    return jobService.resume(username(principal), jobId);
  }

  @PostMapping("/{jobId}/stop")
  @Operation(summary = "Stop a training job")
  public TrainingControlResponse stop(Principal principal, @PathVariable String jobId) {
    return jobService.stop(username(principal), jobId);
  }

  @PostMapping("/{jobId}/reset")
  @Operation(summary = "Reset a training job")
  public TrainingControlResponse reset(Principal principal, @PathVariable String jobId) {
    return jobService.reset(username(principal), jobId);
  }

  @PostMapping("/{jobId}/save")
  @Operation(summary = "Save a training checkpoint")
  public TrainingControlResponse save(Principal principal, @PathVariable String jobId) {
    return jobService.save(username(principal), jobId);
  }

  @ExceptionHandler(IllegalArgumentException.class)
  ResponseEntity<DatasetErrorResponse> trainingBadRequest(IllegalArgumentException ex) {
    return ResponseEntity.status(HttpStatus.BAD_REQUEST)
        .body(new DatasetErrorResponse("BAD_REQUEST", ex.getMessage()));
  }

  private String username(Principal principal) {
    return principal == null ? null : principal.getName();
  }
}
