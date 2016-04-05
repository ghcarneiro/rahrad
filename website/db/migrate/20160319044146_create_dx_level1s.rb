class CreateDxLevel1s < ActiveRecord::Migration
  def change
    create_table :dx_level1s do |t|
      t.string :name
      t.timestamps null: false
    end
  end
end
