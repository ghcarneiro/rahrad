class CreateExpertReports < ActiveRecord::Migration
  def change
    create_table :expert_reports do |t|

      t.timestamps null: false
    end
  end
end
