package com.deepvision.studio.training;

import com.deepvision.studio.training.TrainingDtos.CheckpointTestResult;
import com.deepvision.studio.training.TrainingDtos.DatasetErrorResponse;
import com.deepvision.studio.training.TrainingDtos.DatasetImportResponse;
import com.deepvision.studio.training.TrainingDtos.StartTrainingRequest;
import com.deepvision.studio.training.TrainingDtos.TestCheckpointRequest;
import com.deepvision.studio.training.TrainingDtos.TrainingCheckpointSummary;
import com.deepvision.studio.training.TrainingDtos.TrainingControlResponse;
import com.deepvision.studio.training.TrainingDtos.TrainingDatasetDetail;
import com.deepvision.studio.training.TrainingDtos.TrainingDatasetOption;
import com.deepvision.studio.training.TrainingDtos.TrainingStartResponse;
import com.deepvision.studio.training.TrainingDtos.TrainingStatusResponse;
import com.deepvision.studio.training.TrainingDtos.WeightHistogramResponse;
import jakarta.validation.Valid;
import java.security.Principal;
import java.util.List;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
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
public class TrainingController {
  private final TrainingDatasetService datasetService;
  private final TrainingJobService jobService;

  public TrainingController(TrainingDatasetService datasetService, TrainingJobService jobService) {
    this.datasetService = datasetService;
    this.jobService = jobService;
  }

  @GetMapping("/datasets/builtin")
  public List<TrainingDatasetOption> listBuiltinDatasets() {
    return datasetService.listBuiltin();
  }

  @GetMapping("/datasets/{datasetId}")
  public TrainingDatasetDetail datasetDetail(@PathVariable String datasetId) {
    return datasetService.getDetail(datasetId);
  }

  @GetMapping(value = "/datasets/{datasetId}/preview/{index}", produces = "image/svg+xml")
  public ResponseEntity<String> builtInPreview(@PathVariable String datasetId, @PathVariable int index) {
    return ResponseEntity.ok()
        .header(HttpHeaders.CACHE_CONTROL, "public, max-age=3600")
        .body(datasetService.builtInPreviewSvg(datasetId, index));
  }

  @PostMapping(value = "/datasets/imports", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
  public DatasetImportResponse importDataset(@RequestParam("files") MultipartFile[] files) {
    return datasetService.importDataset(files);
  }

  @PostMapping("/start")
  public TrainingStartResponse start(Principal principal, @Valid @RequestBody StartTrainingRequest request) {
    return jobService.start(request, principal == null ? null : principal.getName());
  }

  @GetMapping("/checkpoints")
  public List<TrainingCheckpointSummary> checkpoints(Principal principal) {
    return jobService.listCheckpoints(principal == null ? null : principal.getName());
  }

  @PostMapping("/checkpoints/{checkpointId}/test")
  public CheckpointTestResult testCheckpoint(
      Principal principal,
      @PathVariable Long checkpointId,
      @Valid @RequestBody TestCheckpointRequest request
  ) {
    return jobService.testCheckpoint(principal == null ? null : principal.getName(), checkpointId, request);
  }

  @GetMapping("/{jobId}/weights/histogram")
  public WeightHistogramResponse histogram(@PathVariable String jobId) {
    return jobService.histogram(jobId);
  }

  @GetMapping("/{jobId}/status")
  public TrainingStatusResponse status(@PathVariable String jobId) {
    return jobService.status(jobId);
  }

  @PostMapping("/{jobId}/pause")
  public TrainingControlResponse pause(@PathVariable String jobId) {
    return jobService.pause(jobId);
  }

  @PostMapping("/{jobId}/resume")
  public TrainingControlResponse resume(@PathVariable String jobId) {
    return jobService.resume(jobId);
  }

  @PostMapping("/{jobId}/stop")
  public TrainingControlResponse stop(@PathVariable String jobId) {
    return jobService.stop(jobId);
  }

  @PostMapping("/{jobId}/reset")
  public TrainingControlResponse reset(@PathVariable String jobId) {
    return jobService.reset(jobId);
  }

  @PostMapping("/{jobId}/save")
  public TrainingControlResponse save(Principal principal, @PathVariable String jobId) {
    return jobService.save(jobId);
  }

  @ExceptionHandler(IllegalArgumentException.class)
  ResponseEntity<DatasetErrorResponse> trainingBadRequest(IllegalArgumentException ex) {
    return ResponseEntity.status(HttpStatus.BAD_REQUEST)
        .body(new DatasetErrorResponse("BAD_REQUEST", ex.getMessage()));
  }
}
