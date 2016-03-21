# encoding: UTF-8
# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# Note that this schema.rb definition is the authoritative source for your
# database schema. If you need to create the application database on another
# system, you should be using db:schema:load, not running all the migrations
# from scratch. The latter is a flawed and unsustainable approach (the more migrations
# you'll amass, the slower it'll run and the greater likelihood for issues).
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema.define(version: 20160321132434) do

  create_table "dx_level1s", force: :cascade do |t|
    t.string   "name"
    t.float    "all_total"
    t.float    "all_good"
    t.float    "all_excellent"
    t.integer  "all_number"
    t.float    "key_total"
    t.float    "key_good"
    t.float    "key_excellent"
    t.integer  "key_number"
    t.float    "cat1_total"
    t.float    "cat1_good"
    t.float    "cat1_excellent"
    t.integer  "cat1_number"
    t.float    "cat2_total"
    t.float    "cat2_good"
    t.float    "cat2_excellent"
    t.integer  "cat2_number"
    t.float    "cat3_total"
    t.float    "cat3_good"
    t.float    "cat3_excellent"
    t.integer  "cat3_number"
    t.datetime "created_at",     null: false
    t.datetime "updated_at",     null: false
  end

  create_table "dx_level2s", force: :cascade do |t|
    t.string   "name"
    t.string   "dx_level1_id"
    t.float    "all_total"
    t.float    "all_good"
    t.float    "all_excellent"
    t.integer  "all_number"
    t.float    "key_total"
    t.float    "key_good"
    t.float    "key_excellent"
    t.integer  "key_number"
    t.float    "cat1_total"
    t.float    "cat1_good"
    t.float    "cat1_excellent"
    t.integer  "cat1_number"
    t.float    "cat2_total"
    t.float    "cat2_good"
    t.float    "cat2_excellent"
    t.integer  "cat2_number"
    t.float    "cat3_total"
    t.float    "cat3_good"
    t.float    "cat3_excellent"
    t.integer  "cat3_number"
    t.datetime "created_at",     null: false
    t.datetime "updated_at",     null: false
  end

  create_table "end_dxes", force: :cascade do |t|
    t.string   "name"
    t.string   "dx_level1_id"
    t.string   "category"
    t.float    "total"
    t.float    "good"
    t.float    "excellent"
    t.integer  "number"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end

  create_table "learner_models", force: :cascade do |t|
    t.float "abdominal"
    t.float "cardiothoracic"
    t.float "ent"
    t.float "neuroradiology"
    t.float "musculoskeletal"
    t.float "paediatric"
    t.float "breast"
    t.float "obsgynae"
    t.float "vascular"
  end

  create_table "reoccurences", force: :cascade do |t|
    t.string   "query"
    t.integer  "interval"
    t.string   "unit"
    t.datetime "time"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.integer  "user_id"
  end

  create_table "saved_searches", force: :cascade do |t|
    t.string   "query"
    t.datetime "time"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.integer  "user_id"
  end

  create_table "users", force: :cascade do |t|
    t.string   "email",                  default: "", null: false
    t.string   "encrypted_password",     default: "", null: false
    t.string   "reset_password_token"
    t.datetime "reset_password_sent_at"
    t.datetime "remember_created_at"
    t.integer  "sign_in_count",          default: 0,  null: false
    t.datetime "current_sign_in_at"
    t.datetime "last_sign_in_at"
    t.string   "current_sign_in_ip"
    t.string   "last_sign_in_ip"
    t.datetime "created_at",                          null: false
    t.datetime "updated_at",                          null: false
  end

  add_index "users", ["email"], name: "index_users_on_email", unique: true
  add_index "users", ["reset_password_token"], name: "index_users_on_reset_password_token", unique: true

end
