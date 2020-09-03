/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "gamma_api.h"

#include <fcntl.h>
#include <sys/stat.h>

#include <chrono>
#include <iostream>
#include <sstream>

#include "api_data/gamma_config.h"
#include "api_data/gamma_doc.h"
#include "api_data/gamma_engine_status.h"
#include "api_data/gamma_response.h"
#include "api_data/gamma_table.h"
#include "gamma_engine.h"
#include "log.h"
#include "utils.h"

INITIALIZE_EASYLOGGINGPP

static bool log_dir_flag = false;

int SetLogDictionary(const std::string &log_dir);

void *Init(const char *config_str, int len) {
  tig_gamma::Config config;
  config.Deserialize(config_str, len);

  if (not log_dir_flag) {
    const std::string &log_dir = config.LogDir();
    SetLogDictionary(log_dir);
    log_dir_flag = true;
  }

  const std::string &path = config.Path();
  tig_gamma::GammaEngine *engine = tig_gamma::GammaEngine::GetInstance(path);
  if (engine == nullptr) {
    LOG(ERROR) << "Engine init faild!";
    return nullptr;
  }
  LOG(INFO) << "Engine init successed!";
  return static_cast<void *>(engine);
}

int SetLogDictionary(const std::string &log_dir) {
  const std::string &dir = log_dir;
  if (!utils::isFolderExist(dir.c_str())) {
    mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

  el::Configurations defaultConf;
  // To set GLOBAL configurations you may use
  el::Loggers::addFlag(el::LoggingFlag::StrictLogFileSizeCheck);
  defaultConf.setGlobally(el::ConfigurationType::Format,
                          "%level %datetime %fbase:%line %msg");
  defaultConf.setGlobally(el::ConfigurationType::ToFile, "true");
  defaultConf.setGlobally(el::ConfigurationType::ToStandardOutput, "false");
  defaultConf.setGlobally(el::ConfigurationType::MaxLogFileSize,
                          "209715200");  // 200MB
  defaultConf.setGlobally(el::ConfigurationType::Filename, dir + "/gamma.log");
  el::Loggers::reconfigureLogger("default", defaultConf);
  el::Helpers::installPreRollOutCallback(
      [](const char *filename, std::size_t size) {
        // SHOULD NOT LOG ANYTHING HERE BECAUSE LOG FILE IS CLOSED!
        std::cout << "************** Rolling out [" << filename
                  << "] because it reached [" << size << " bytes]" << std::endl;
        std::time_t t = std::time(nullptr);
        char mbstr[100];
        if (std::strftime(mbstr, sizeof(mbstr), "%F-%T", std::localtime(&t))) {
          std::cout << mbstr << '\n';
        }
        std::stringstream ss;
        ss << "mv " << filename << " " << filename << "-" << mbstr;
        system(ss.str().c_str());
      });

  LOG(INFO) << "Version [" << GIT_SHA1 << "]";
  return 0;
}

int Close(void *engine) {
  LOG(INFO) << "Close";
  delete static_cast<tig_gamma::GammaEngine *>(engine);
  return 0;
}

int CreateTable(void *engine, const char *table_str, int len) {
  tig_gamma::TableInfo table;
  table.Deserialize(table_str, len);
  int ret = static_cast<tig_gamma::GammaEngine *>(engine)->CreateTable(table);
  return ret;
}

int AddOrUpdateDoc(void *engine, const char *doc_str, int len) {
  tig_gamma::Doc doc;
  doc.Deserialize(doc_str, len);
  return static_cast<tig_gamma::GammaEngine *>(engine)->AddOrUpdate(doc);
}

int UpdateDoc(void *engine, const char *doc_str, int len) {
  tig_gamma::Doc doc;
  doc.Deserialize(doc_str, len);
  return static_cast<tig_gamma::GammaEngine *>(engine)->Update(&doc);
}

int Search(void *engine, const char *request_str, int req_len,
           char **response_str, int *res_len) {
  tig_gamma::Response response;
  tig_gamma::Request request;
  request.Deserialize(request_str, req_len);

  int ret =
      static_cast<tig_gamma::GammaEngine *>(engine)->Search(request, response);

  response.Serialize(response_str, res_len);

  return ret;
}

int DeleteDoc(void *engine, const char *docid, int docid_len) {
  std::string id = std::string(docid, docid_len);
  int ret = static_cast<tig_gamma::GammaEngine *>(engine)->Delete(id);
  return ret;
}

int GetDocByID(void *engine, const char *docid, int docid_len, char **doc_str,
               int *len) {
  tig_gamma::Doc doc;
  std::string id = std::string(docid, docid_len);
  int ret = static_cast<tig_gamma::GammaEngine *>(engine)->GetDoc(id, doc);
  doc.Serialize(doc_str, len);
  return ret;
}

int BuildIndex(void *engine) {
  int ret = static_cast<tig_gamma::GammaEngine *>(engine)->BuildIndex();
  return ret;
}

void GetEngineStatus(void *engine, char **status_str, int *len) {
  tig_gamma::EngineStatus engine_status;
  static_cast<tig_gamma::GammaEngine *>(engine)->GetIndexStatus(engine_status);
  engine_status.Serialize(status_str, len);
}

int Dump(void *engine) {
  int ret = static_cast<tig_gamma::GammaEngine *>(engine)->Dump();
  return ret;
}

int Load(void *engine) {
  int ret = static_cast<tig_gamma::GammaEngine *>(engine)->Load();
  return ret;
}

int DelDocByQuery(void *engine, const char *request_str, int len) {
  tig_gamma::Request request;
  request.Deserialize(request_str, len);
  int ret =
      static_cast<tig_gamma::GammaEngine *>(engine)->DelDocByQuery(request);
  return ret;
}
